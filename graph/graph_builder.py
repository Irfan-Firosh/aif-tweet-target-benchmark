"""
Graph construction for Tweet Target Entity & Sentiment Detection.

Builds a heterogeneous directed multigraph modeling relationships between
tweets, target entities, authors, events, and parties for the dev and test splits.

Usage:
    python -m graph.graph_builder
"""

import json
import logging
import pickle
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from data_collector import load_json, load_party_mapping
from graph.graph_config import (
    AUTHOR_PREFIX,
    EDGE_ABOUT,
    EDGE_AUTHORED,
    EDGE_BELONGS_TO,
    EDGE_CO_TARGETED,
    EDGE_MENTIONS,
    EDGE_TARGETS,
    ENTITY_PREFIX,
    EVENT_PREFIX,
    NODE_TYPE_AUTHOR,
    NODE_TYPE_ENTITY,
    NODE_TYPE_EVENT,
    NODE_TYPE_PARTY,
    NODE_TYPE_TWEET,
    PARTY_PREFIX,
    TWEET_PREFIX,
    infer_event,
    status_id_to_datetime,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_screen_name(tweet_url: str) -> str | None:
    """Extract screen name from tweet URL (handles www.twitter.com)."""
    parts = tweet_url.rstrip("/").split("/")
    for i, part in enumerate(parts):
        if "twitter.com" in part and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _extract_date_str(tweet_url: str) -> str:
    """Extract ISO date string from tweet URL status ID."""
    parts = tweet_url.rstrip("/").split("/")
    try:
        dt = status_id_to_datetime(parts[-1])
        return dt.strftime("%Y-%m-%d")
    except (ValueError, IndexError, OverflowError):
        return ""


def build_graph(splits: list[str] | None = None) -> nx.MultiDiGraph:
    """
    Build a heterogeneous directed multigraph from the tweet target dataset.

    Args:
        splits: Which splits to include. Defaults to ["dev", "test"].

    Returns:
        A NetworkX MultiDiGraph with typed nodes and edges.
    """
    if splits is None:
        splits = ["dev", "test"]

    raw = load_json(config.TARGET_SPLIT_FILE)
    party_map = load_party_mapping()

    G = nx.MultiDiGraph()

    # Add party nodes
    for party_name in ["Democrat", "Republican"]:
        pid = f"{PARTY_PREFIX}{party_name}"
        G.add_node(pid, node_type=NODE_TYPE_PARTY, label=party_name)

    events_seen = set()

    for split in splits:
        if split not in raw:
            logger.warning(f"Split '{split}' not found in dataset")
            continue

        for tweet_url, annotations in raw[split].items():
            tweet_id = f"{TWEET_PREFIX}{tweet_url}"
            screen_name = extract_screen_name(tweet_url)
            event_name = infer_event(tweet_url)
            tweet_date = _extract_date_str(tweet_url)

            # --- Tweet node ---
            if tweet_id not in G:
                G.add_node(
                    tweet_id,
                    node_type=NODE_TYPE_TWEET,
                    url=tweet_url,
                    screen_name=screen_name or "",
                    date=tweet_date,
                    event=event_name or "unknown",
                    splits=split,
                )
            else:
                # Tweet appears in multiple splits — update splits attribute
                existing = G.nodes[tweet_id].get("splits", "")
                if split not in existing.split(","):
                    G.nodes[tweet_id]["splits"] = f"{existing},{split}"

            # --- Author node + edges ---
            if screen_name:
                author_id = f"{AUTHOR_PREFIX}{screen_name}"
                if author_id not in G:
                    party_code = party_map.get(f"@{screen_name}") or party_map.get(screen_name)
                    party_label = ""
                    if party_code in ("D", "Democrat"):
                        party_label = "Democrat"
                    elif party_code in ("R", "Republican"):
                        party_label = "Republican"

                    G.add_node(
                        author_id,
                        node_type=NODE_TYPE_AUTHOR,
                        screen_name=screen_name,
                        party=party_label or "unknown",
                    )

                    # BELONGS_TO edge
                    if party_label:
                        G.add_edge(
                            author_id,
                            f"{PARTY_PREFIX}{party_label}",
                            edge_type=EDGE_BELONGS_TO,
                        )

                # AUTHORED edge (one per author-tweet pair)
                if not G.has_edge(author_id, tweet_id):
                    G.add_edge(author_id, tweet_id, edge_type=EDGE_AUTHORED)

            # --- Event node + ABOUT edge ---
            if event_name:
                event_id = f"{EVENT_PREFIX}{event_name}"
                events_seen.add(event_name)
                if event_id not in G:
                    G.add_node(event_id, node_type=NODE_TYPE_EVENT, label=event_name)
                if not G.has_edge(tweet_id, event_id):
                    G.add_edge(tweet_id, event_id, edge_type=EDGE_ABOUT, split=split)

            # --- Entity nodes + TARGETS/MENTIONS edges ---
            targets_in_tweet = []
            for ann in annotations:
                entity_name = ann["choice"]
                entity_id = f"{ENTITY_PREFIX}{entity_name}"

                if entity_id not in G:
                    G.add_node(entity_id, node_type=NODE_TYPE_ENTITY, label=entity_name)

                is_target = ann["label"] == 1
                sentiment = ann.get("sentiment", "")

                if is_target:
                    G.add_edge(
                        tweet_id,
                        entity_id,
                        edge_type=EDGE_TARGETS,
                        sentiment=sentiment or "unknown",
                        split=split,
                    )
                    targets_in_tweet.append(entity_name)
                else:
                    G.add_edge(
                        tweet_id,
                        entity_id,
                        edge_type=EDGE_MENTIONS,
                        split=split,
                    )

            # --- CO_TARGETED edges ---
            for e1, e2 in combinations(sorted(targets_in_tweet), 2):
                G.add_edge(
                    f"{ENTITY_PREFIX}{e1}",
                    f"{ENTITY_PREFIX}{e2}",
                    edge_type=EDGE_CO_TARGETED,
                    tweet_url=tweet_url,
                    split=split,
                )

    logger.info(
        f"Built graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, events={events_seen}"
    )
    return G


def save_graph(G: nx.MultiDiGraph, output_dir: Path, prefix: str = "tweet_target") -> None:
    """Save graph in GraphML, pickle, and JSON formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # GraphML
    graphml_path = output_dir / f"{prefix}_graph.graphml"
    nx.write_graphml(G, str(graphml_path))
    logger.info(f"Saved GraphML: {graphml_path}")

    # Pickle
    pickle_path = output_dir / f"{prefix}_graph.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(G, f)
    logger.info(f"Saved pickle: {pickle_path}")

    # JSON node-link
    json_path = output_dir / f"{prefix}_graph.json"
    data = json_graph.node_link_data(G)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON: {json_path}")


def print_graph_stats(G: nx.MultiDiGraph) -> dict:
    """Print and return comprehensive graph statistics."""
    node_types = Counter(G.nodes[n].get("node_type", "unknown") for n in G.nodes())
    edge_types = Counter(G.edges[e].get("edge_type", "unknown") for e in G.edges(keys=True))

    sentiment_dist = Counter(
        G.edges[e].get("sentiment", "unknown")
        for e in G.edges(keys=True)
        if G.edges[e].get("edge_type") == EDGE_TARGETS
    )

    degree_by_type = defaultdict(list)
    for node in G.nodes():
        degree_by_type[G.nodes[node].get("node_type", "unknown")].append(G.degree(node))

    party_dist = Counter(
        G.nodes[n].get("party", "unknown")
        for n in G.nodes()
        if G.nodes[n].get("node_type") == NODE_TYPE_AUTHOR
    )

    event_dist = Counter(
        G.nodes[n].get("event", "unknown")
        for n in G.nodes()
        if G.nodes[n].get("node_type") == NODE_TYPE_TWEET
    )

    entity_degrees = [
        (G.nodes[n].get("label", n), G.degree(n))
        for n in G.nodes()
        if G.nodes[n].get("node_type") == NODE_TYPE_ENTITY
    ]
    top_entities = sorted(entity_degrees, key=lambda x: -x[1])[:10]

    author_tweet_counts = []
    for n in G.nodes():
        if G.nodes[n].get("node_type") == NODE_TYPE_AUTHOR:
            count = sum(
                1 for _, _, d in G.out_edges(n, data=True)
                if d.get("edge_type") == EDGE_AUTHORED
            )
            author_tweet_counts.append((G.nodes[n].get("screen_name", n), count))
    top_authors = sorted(author_tweet_counts, key=lambda x: -x[1])[:10]

    print(f"\n{'=' * 60}")
    print("  Graph Statistics (dev + test)")
    print(f"{'=' * 60}")

    print("\n--- Node Counts ---")
    for ntype, count in sorted(node_types.items()):
        degrees = degree_by_type[ntype]
        avg_deg = sum(degrees) / len(degrees) if degrees else 0
        print(f"  {ntype:>10}: {count:>5}  (avg degree: {avg_deg:.1f})")
    print(f"  {'TOTAL':>10}: {G.number_of_nodes():>5}")

    print("\n--- Edge Counts ---")
    for etype, count in sorted(edge_types.items()):
        print(f"  {etype:>15}: {count:>5}")
    print(f"  {'TOTAL':>15}: {G.number_of_edges():>5}")

    print("\n--- Sentiment Distribution (TARGETS edges) ---")
    for s, c in sorted(sentiment_dist.items()):
        print(f"  {s:>10}: {c:>5}")

    print("\n--- Author Party Distribution ---")
    for p, c in sorted(party_dist.items()):
        print(f"  {p:>12}: {c:>5}")

    print("\n--- Event Distribution (tweets) ---")
    for ev, c in sorted(event_dist.items()):
        print(f"  {ev}: {c}")

    print("\n--- Top 10 Most Connected Entities ---")
    for label, deg in top_entities:
        print(f"  {deg:>4}  {label}")

    print("\n--- Top 10 Most Prolific Authors ---")
    for sn, tc in top_authors:
        print(f"  {tc:>4} tweets  @{sn}")

    return {
        "node_counts": dict(node_types),
        "edge_counts": dict(edge_types),
        "sentiment_dist": dict(sentiment_dist),
        "party_dist": dict(party_dist),
        "event_dist": dict(event_dist),
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
    }


if __name__ == "__main__":
    logger.info("=== Graph Construction: Tweet Target Entity & Sentiment ===")

    G = build_graph(splits=["dev", "test"])
    stats = print_graph_stats(G)

    graph_output_dir = config.OUTPUT_DIR / "graph"
    save_graph(G, graph_output_dir)

    stats_path = graph_output_dir / "graph_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")

    logger.info("Done.")
