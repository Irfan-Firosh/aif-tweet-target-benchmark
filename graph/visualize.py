"""
Interactive graph visualization using pyvis.

Generates an HTML file you can open in a browser with zoom, pan, and drag.

Usage:
    python -m graph.visualize
"""

import pickle
import sys
from pathlib import Path

from pyvis.network import Network

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from graph.graph_config import (
    EDGE_ABOUT,
    EDGE_AUTHORED,
    EDGE_BELONGS_TO,
    EDGE_CO_TARGETED,
    EDGE_MENTIONS,
    EDGE_TARGETS,
    NODE_TYPE_AUTHOR,
    NODE_TYPE_ENTITY,
    NODE_TYPE_EVENT,
    NODE_TYPE_PARTY,
    NODE_TYPE_TWEET,
)

# Node colors and sizes by type
NODE_STYLE = {
    NODE_TYPE_TWEET:  {"color": "#4CAF50", "size": 8,  "shape": "dot"},
    NODE_TYPE_ENTITY: {"color": "#FF5722", "size": 15, "shape": "dot"},
    NODE_TYPE_AUTHOR: {"color": "#2196F3", "size": 12, "shape": "dot"},
    NODE_TYPE_EVENT:  {"color": "#9C27B0", "size": 30, "shape": "star"},
    NODE_TYPE_PARTY:  {"color": "#FF9800", "size": 25, "shape": "diamond"},
}

# Edge colors by type
EDGE_STYLE = {
    EDGE_TARGETS:     {"color": "#E53935", "width": 1.5},
    EDGE_MENTIONS:    {"color": "#BDBDBD", "width": 0.5},
    EDGE_AUTHORED:    {"color": "#1E88E5", "width": 1.0},
    EDGE_ABOUT:       {"color": "#8E24AA", "width": 1.0},
    EDGE_BELONGS_TO:  {"color": "#FB8C00", "width": 2.0},
    EDGE_CO_TARGETED: {"color": "#FDD835", "width": 0.5},
}


def build_label(node_id: str, attrs: dict) -> str:
    """Build a short display label for a node."""
    ntype = attrs.get("node_type", "")
    if ntype == NODE_TYPE_TWEET:
        return f"@{attrs.get('screen_name', '?')} ({attrs.get('date', '')})"
    if ntype == NODE_TYPE_AUTHOR:
        return f"@{attrs.get('screen_name', node_id)}"
    if ntype == NODE_TYPE_ENTITY:
        return attrs.get("label", node_id)
    if ntype == NODE_TYPE_EVENT:
        return attrs.get("label", node_id)
    if ntype == NODE_TYPE_PARTY:
        return attrs.get("label", node_id)
    return str(node_id)


def build_title(node_id: str, attrs: dict) -> str:
    """Build a hover tooltip for a node."""
    lines = [f"<b>{attrs.get('node_type', 'unknown').upper()}</b>"]
    for k, v in attrs.items():
        if k != "node_type":
            lines.append(f"{k}: {v}")
    return "<br>".join(lines)


def visualize_full(G, output_path: Path) -> None:
    """Render the full graph as an interactive HTML file."""
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
        notebook=False,
    )

    # Physics settings for better layout
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.01,
        damping=0.09,
    )

    # Add nodes
    for node_id, attrs in G.nodes(data=True):
        ntype = attrs.get("node_type", "unknown")
        style = NODE_STYLE.get(ntype, {"color": "#999", "size": 10, "shape": "dot"})
        net.add_node(
            node_id,
            label=build_label(node_id, attrs),
            title=build_title(node_id, attrs),
            color=style["color"],
            size=style["size"],
            shape=style["shape"],
        )

    # Add edges
    for u, v, key, attrs in G.edges(data=True, keys=True):
        etype = attrs.get("edge_type", "unknown")
        style = EDGE_STYLE.get(etype, {"color": "#666", "width": 0.5})
        title_parts = [etype]
        if "sentiment" in attrs and attrs["sentiment"]:
            title_parts.append(f"sentiment: {attrs['sentiment']}")
        if "split" in attrs:
            title_parts.append(f"split: {attrs['split']}")
        net.add_edge(
            u, v,
            title="<br>".join(title_parts),
            color=style["color"],
            width=style["width"],
        )

    net.show_buttons(filter_=["physics"])
    net.save_graph(str(output_path))
    print(f"Full graph saved to: {output_path}")


def visualize_no_tweets(G, output_path: Path) -> None:
    """
    Render a condensed view without tweet nodes.
    Shows author → entity (via targets) and entity ↔ entity (co-targeted).
    """
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
        notebook=False,
    )
    net.barnes_hut(
        gravity=-5000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.005,
        damping=0.09,
    )

    # Collect author→entity targeting relationships (collapse through tweets)
    from collections import defaultdict, Counter
    author_targets = defaultdict(Counter)  # author_id → {entity_id: count}
    author_sentiments = defaultdict(lambda: defaultdict(Counter))  # author→entity→sentiment_counts

    for u, v, attrs in G.edges(data=True):
        if attrs.get("edge_type") == EDGE_TARGETS:
            tweet_attrs = G.nodes.get(u, {})
            screen_name = tweet_attrs.get("screen_name", "")
            if screen_name:
                author_id = f"author::{screen_name}"
                author_targets[author_id][v] += 1
                sentiment = attrs.get("sentiment", "unknown")
                author_sentiments[author_id][v][sentiment] += 1

    # Add non-tweet nodes
    for node_id, attrs in G.nodes(data=True):
        ntype = attrs.get("node_type", "unknown")
        if ntype == NODE_TYPE_TWEET:
            continue
        style = NODE_STYLE.get(ntype, {"color": "#999", "size": 10, "shape": "dot"})
        # Scale entity size by degree
        size = style["size"]
        if ntype == NODE_TYPE_ENTITY:
            size = max(8, min(40, G.degree(node_id) // 3))
        net.add_node(
            node_id,
            label=build_label(node_id, attrs),
            title=build_title(node_id, attrs),
            color=style["color"],
            size=size,
            shape=style["shape"],
        )

    # Add author → entity edges (collapsed)
    for author_id, entity_counts in author_targets.items():
        for entity_id, count in entity_counts.items():
            sentiments = author_sentiments[author_id][entity_id]
            top_sentiment = sentiments.most_common(1)[0][0] if sentiments else "unknown"
            color = {"positive": "#43A047", "negative": "#E53935", "neutral": "#FDD835"}.get(
                top_sentiment, "#999"
            )
            net.add_edge(
                author_id, entity_id,
                title=f"targets ({count}x)<br>dominant sentiment: {top_sentiment}",
                color=color,
                width=min(4, 0.5 + count * 0.5),
            )

    # Add non-tweet edges (BELONGS_TO, CO_TARGETED)
    co_target_counts = Counter()
    for u, v, attrs in G.edges(data=True):
        etype = attrs.get("edge_type", "")
        if etype == EDGE_BELONGS_TO:
            style = EDGE_STYLE[EDGE_BELONGS_TO]
            net.add_edge(u, v, title=etype, color=style["color"], width=style["width"])
        elif etype == EDGE_CO_TARGETED:
            pair = tuple(sorted([u, v]))
            co_target_counts[pair] += 1

    for (e1, e2), count in co_target_counts.items():
        if count >= 2:  # Only show co-targeting with frequency >= 2
            net.add_edge(
                e1, e2,
                title=f"co-targeted {count}x",
                color="#FDD835",
                width=min(3, 0.3 + count * 0.3),
            )

    net.show_buttons(filter_=["physics"])
    net.save_graph(str(output_path))
    print(f"Condensed graph (no tweets) saved to: {output_path}")


if __name__ == "__main__":
    graph_dir = config.OUTPUT_DIR / "graph"
    pkl_path = graph_dir / "tweet_target_graph.pkl"

    if not pkl_path.exists():
        print("Graph pickle not found. Run `python -m graph.graph_builder` first.")
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        G = pickle.load(f)

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Full graph visualization
    visualize_full(G, graph_dir / "graph_full.html")

    # Condensed view (no tweet nodes, author→entity collapsed)
    visualize_no_tweets(G, graph_dir / "graph_condensed.html")

    print("\nOpen the HTML files in your browser to explore the graph.")
