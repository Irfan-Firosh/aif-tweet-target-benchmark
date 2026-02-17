# Graph Module Guide

In-depth walkthrough of every file in `graph/`, how the code works, and what the resulting graph represents.

---

## Table of Contents

1. [Directory Overview](#directory-overview)
2. [graph/\_\_init\_\_.py](#graph__init__py)
3. [graph/graph_config.py](#graphgraph_configpy)
   - [Twitter Snowflake Decoding](#twitter-snowflake-decoding)
   - [Event Date Ranges](#event-date-ranges)
   - [Node and Edge Constants](#node-and-edge-constants)
   - [Functions](#graph_config-functions)
4. [graph/graph_builder.py](#graphgraph_builderpy)
   - [Imports and Path Setup](#imports-and-path-setup)
   - [extract_screen_name()](#extract_screen_name)
   - [_extract_date_str()](#_extract_date_str)
   - [build_graph() — The Core Algorithm](#build_graph--the-core-algorithm)
   - [save_graph()](#save_graph)
   - [print_graph_stats()](#print_graph_stats)
   - [Main Block](#main-block)
5. [graph/visualize.py](#graphvisualizepy)
   - [Styling Constants](#styling-constants)
   - [visualize_full()](#visualize_full)
   - [visualize_no_tweets()](#visualize_no_tweets--the-condensed-view)
6. [The Resulting Graph](#the-resulting-graph)
   - [Node Types and Their Attributes](#node-types-and-their-attributes)
   - [Edge Types and Their Attributes](#edge-types-and-their-attributes)
   - [Graph Statistics](#graph-statistics)
7. [Output Files](#output-files)
8. [How to Run](#how-to-run)

---

## Directory Overview

```
graph/
  __init__.py        # Package marker (1 line)
  graph_config.py    # Constants, event inference, ID formatting
  graph_builder.py   # Core: loads data, builds graph, saves, prints stats
  visualize.py       # Interactive HTML visualizations via pyvis
  GUIDE.md           # This file
```

The module depends on two project-root files it imports:
- `config.py` — provides file path constants (`TARGET_SPLIT_FILE`, `OUTPUT_DIR`, etc.)
- `data_collector.py` — provides `load_json()` and `load_party_mapping()`

External dependencies: `networkx`, `pyvis`.

---

## graph/\_\_init\_\_.py

```python
"""Graph construction for Tweet Target Entity & Sentiment Detection."""
```

Just a package marker. It contains only a docstring so that Python recognizes `graph/` as an importable package. This is needed because `graph_builder.py` does `from graph.graph_config import ...` — without `__init__.py`, that import would fail.

---

## graph/graph_config.py

This file holds all the constants and two utility functions. Nothing here touches the dataset — it's pure configuration.

### Twitter Snowflake Decoding

```python
TWITTER_EPOCH_OFFSET = 1288834974657  # milliseconds
```

Twitter uses "snowflake IDs" for tweet status IDs. These are not random — they encode a timestamp. The formula to extract it is:

```
timestamp_ms = (status_id >> 22) + 1288834974657
```

- `>> 22` is a bitwise right-shift by 22 bits. The snowflake format packs the timestamp in the upper 41 bits, with 22 lower bits used for machine/sequence info. Shifting right by 22 strips those lower bits, leaving just the timestamp portion.
- `1288834974657` is Twitter's custom epoch in milliseconds (Nov 4, 2010 01:42:54.657 UTC). Twitter doesn't count from the Unix epoch (Jan 1, 1970) — they count from their own start date. Adding this offset converts to a standard Unix timestamp.

**Why this matters:** The raw dataset (`target_task_split_v7.json`) does NOT store which political event a tweet belongs to. The `event` field is always null. But we know:
- Kavanaugh tweets were posted Sep–Oct 2018
- George Floyd tweets were posted May–Nov 2020
- Capitol Attack tweets were posted Jan–Mar 2021

So by decoding the timestamp from the tweet URL's status ID, we can reliably determine which event each tweet belongs to.

### Event Date Ranges

```python
EVENT_DATE_RANGES = {
    "Kavanaugh Supreme Court Nomination": {
        "start": datetime(2018, 9, 1, tzinfo=timezone.utc),
        "end":   datetime(2018, 11, 1, tzinfo=timezone.utc),
    },
    "Death of George Floyd": {
        "start": datetime(2020, 5, 1, tzinfo=timezone.utc),
        "end":   datetime(2020, 12, 1, tzinfo=timezone.utc),
    },
    "2021 US Capitol Attack": {
        "start": datetime(2021, 1, 1, tzinfo=timezone.utc),
        "end":   datetime(2021, 4, 1, tzinfo=timezone.utc),
    },
}
```

Three non-overlapping windows. Every tweet in the dataset falls cleanly into one of these. The ranges are intentionally generous (e.g., Floyd goes to Dec 2020) to capture the tail end of the discourse.

### Node and Edge Constants

**Node ID prefixes** prevent collisions in the graph. For example, a politician named "George Floyd" (hypothetically) would be `author::GeorgeFloyd` while the entity would be `entity::George Floyd`. Without prefixes, they'd collide into a single node.

```python
TWEET_PREFIX  = "tweet::"    # e.g. "tweet::https://www.twitter.com/..."
ENTITY_PREFIX = "entity::"   # e.g. "entity::George Floyd"
AUTHOR_PREFIX = "author::"   # e.g. "author::LacyClayMO1"
EVENT_PREFIX  = "event::"    # e.g. "event::Death of George Floyd"
PARTY_PREFIX  = "party::"    # e.g. "party::Democrat"
```

**Node type labels** are stored as the `node_type` attribute on every node:

```python
NODE_TYPE_TWEET  = "tweet"
NODE_TYPE_ENTITY = "entity"
NODE_TYPE_AUTHOR = "author"
NODE_TYPE_EVENT  = "event"
NODE_TYPE_PARTY  = "party"
```

**Edge type labels** are stored as the `edge_type` attribute on every edge:

```python
EDGE_AUTHORED    = "AUTHORED"      # author wrote this tweet
EDGE_TARGETS     = "TARGETS"       # tweet targets this entity (label=1)
EDGE_MENTIONS    = "MENTIONS"      # entity is a candidate but NOT a target (label=0)
EDGE_BELONGS_TO  = "BELONGS_TO"    # author is a member of this party
EDGE_ABOUT       = "ABOUT"         # tweet is about this event
EDGE_CO_TARGETED = "CO_TARGETED"   # two entities were both targeted in the same tweet
```

### graph_config Functions

**`status_id_to_datetime(status_id)`**

Takes a status ID string (e.g., `"1269346232650149891"`), applies the snowflake formula, and returns a timezone-aware UTC `datetime` object.

```python
def status_id_to_datetime(status_id: str) -> datetime:
    ts_ms = (int(status_id) >> 22) + TWITTER_EPOCH_OFFSET
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
```

Step by step for ID `1269346232650149891`:
1. `int("1269346232650149891")` = 1269346232650149891
2. `>> 22` = 302582838125 (strip lower 22 bits)
3. `+ 1288834974657` = 1591417812782 (Unix timestamp in ms)
4. `/ 1000` = 1591417812.782 (Unix timestamp in seconds)
5. `datetime.fromtimestamp(...)` = 2020-06-06 03:30:12 UTC

This falls within the George Floyd range, so the tweet is classified as belonging to that event.

**`infer_event(tweet_url)`**

Extracts the status ID from the URL, converts to datetime, then checks which event range it falls into.

```python
def infer_event(tweet_url: str) -> str | None:
    parts = tweet_url.rstrip("/").split("/")
    status_id = parts[-1]  # Last segment of URL is the status ID
    ...
    for event_name, date_range in EVENT_DATE_RANGES.items():
        if date_range["start"] <= dt <= date_range["end"]:
            return event_name
    return None
```

Given `https://www.twitter.com/AlLawsonJr/statuses/1269346232650149891`:
- `parts[-1]` = `"1269346232650149891"`
- Decoded datetime = 2020-06-06
- Falls within "Death of George Floyd" range
- Returns `"Death of George Floyd"`

---

## graph/graph_builder.py

This is the main script. It reads the raw dataset, constructs a NetworkX graph, prints statistics, and saves to multiple formats.

### Imports and Path Setup

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

This line adds the project root to `sys.path`. Since the script lives in `graph/`, it needs to go one level up to find `config.py` and `data_collector.py`. Without this, `import config` would fail.

Key imports from the project:
- `config.TARGET_SPLIT_FILE` — path to `data/tweet_target_data/data/target_task_split_v7.json`
- `config.OUTPUT_DIR` — path to `output/`
- `data_collector.load_json()` — reads a JSON file and returns a Python dict/list
- `data_collector.load_party_mapping()` — reads `remaining_party_mapping.json` and returns `{"@screen_name": "D"/"R", ...}`

Key imports from NetworkX:
- `nx.MultiDiGraph` — a directed graph that allows multiple parallel edges between the same pair of nodes. "Multi" = parallel edges allowed, "Di" = directed (edges have a source and target).

### extract_screen_name()

```python
def extract_screen_name(tweet_url: str) -> str | None:
    parts = tweet_url.rstrip("/").split("/")
    for i, part in enumerate(parts):
        if "twitter.com" in part and i + 1 < len(parts):
            return parts[i + 1]
    return None
```

Given `https://www.twitter.com/AlLawsonJr/statuses/1269346232650149891`:
1. `split("/")` produces `["https:", "", "www.twitter.com", "AlLawsonJr", "statuses", "1269346232650149891"]`
2. Iterates through parts, finds `"www.twitter.com"` (which contains `"twitter.com"`)
3. Returns the next part: `"AlLawsonJr"`

**Why not reuse `data_collector.extract_screen_name()`?** The original function in `data_collector.py` uses `parts.index("twitter.com")`, which looks for an **exact match**. But the URL splits into `"www.twitter.com"`, not `"twitter.com"`, so the original function always returns `None`. This is why all entries in `output/tweet_target_raw.json` have `author: null`. Our version uses a substring check (`"twitter.com" in part`) which handles the `www.` prefix correctly.

### _extract_date_str()

```python
def _extract_date_str(tweet_url: str) -> str:
    parts = tweet_url.rstrip("/").split("/")
    try:
        dt = status_id_to_datetime(parts[-1])
        return dt.strftime("%Y-%m-%d")
    except (ValueError, IndexError, OverflowError):
        return ""
```

A thin wrapper around `status_id_to_datetime()` that grabs the status ID from the URL and formats the date as `"YYYY-MM-DD"`. Used to store a human-readable date on tweet nodes. The `try/except` handles malformed URLs gracefully.

### build_graph() — The Core Algorithm

This is the heart of the module. Here's the step-by-step logic:

**Step 1: Load raw data**

```python
raw = load_json(config.TARGET_SPLIT_FILE)
party_map = load_party_mapping()
```

- `raw` is a nested dict: `{"train": {...}, "dev": {...}, "test": {...}}`
- Each split maps `tweet_url → list of annotations`
- Each annotation is `{"choice": "entity name", "label": 0 or 1, "sentiment": "positive"/"negative"/"neutral"}`
- `party_map` is `{"@screen_name": "D" or "R", ...}` for 57 politicians

**Step 2: Create the empty graph and seed party nodes**

```python
G = nx.MultiDiGraph()
for party_name in ["Democrat", "Republican"]:
    pid = f"{PARTY_PREFIX}{party_name}"
    G.add_node(pid, node_type=NODE_TYPE_PARTY, label=party_name)
```

We pre-create the two party nodes because they'll be referenced by author BELONGS_TO edges later. `nx.MultiDiGraph()` creates an empty directed multigraph.

**Step 3: Iterate over each split and each tweet**

```python
for split in splits:              # ["dev", "test"]
    for tweet_url, annotations in raw[split].items():
```

For each tweet URL in the split, we process it through several stages:

**Step 3a: Create the tweet node**

```python
tweet_id = f"{TWEET_PREFIX}{tweet_url}"
screen_name = extract_screen_name(tweet_url)
event_name = infer_event(tweet_url)
tweet_date = _extract_date_str(tweet_url)

if tweet_id not in G:
    G.add_node(tweet_id, node_type=NODE_TYPE_TWEET, url=tweet_url,
               screen_name=screen_name or "", date=tweet_date,
               event=event_name or "unknown", splits=split)
else:
    # Same tweet appears in multiple splits — append to splits attribute
    existing = G.nodes[tweet_id].get("splits", "")
    if split not in existing.split(","):
        G.nodes[tweet_id]["splits"] = f"{existing},{split}"
```

The `if tweet_id not in G` check is important because some tweet URLs appear in both dev and test with different entity annotations. When that happens, we don't create a duplicate node — we just update the `splits` attribute to `"dev,test"`.

**Step 3b: Create the author node and its edges**

```python
if screen_name:
    author_id = f"{AUTHOR_PREFIX}{screen_name}"
    if author_id not in G:
        party_code = party_map.get(f"@{screen_name}") or party_map.get(screen_name)
        party_label = ""
        if party_code in ("D", "Democrat"):
            party_label = "Democrat"
        elif party_code in ("R", "Republican"):
            party_label = "Republican"

        G.add_node(author_id, node_type=NODE_TYPE_AUTHOR,
                   screen_name=screen_name, party=party_label or "unknown")

        if party_label:
            G.add_edge(author_id, f"{PARTY_PREFIX}{party_label}",
                       edge_type=EDGE_BELONGS_TO)

    if not G.has_edge(author_id, tweet_id):
        G.add_edge(author_id, tweet_id, edge_type=EDGE_AUTHORED)
```

For each author (identified by screen name from the URL):
1. Check if we've already created a node for this author (they may have written many tweets)
2. If new, look up their party in `remaining_party_mapping.json`. The file uses `"@screen_name"` keys with `"D"`/`"R"` values, so we try both `"@LacyClayMO1"` and `"LacyClayMO1"`.
3. If party is found, create a BELONGS_TO edge from author to the party node
4. Create an AUTHORED edge from author to tweet (once per unique pair)

The `G.has_edge()` check prevents duplicate AUTHORED edges if the same tweet-URL appears in multiple splits.

**Step 3c: Create the event node and ABOUT edge**

```python
if event_name:
    event_id = f"{EVENT_PREFIX}{event_name}"
    events_seen.add(event_name)
    if event_id not in G:
        G.add_node(event_id, node_type=NODE_TYPE_EVENT, label=event_name)
    if not G.has_edge(tweet_id, event_id):
        G.add_edge(tweet_id, event_id, edge_type=EDGE_ABOUT, split=split)
```

If `infer_event()` successfully decoded the timestamp and matched a date range, we create the event node (if not already present) and add an ABOUT edge from the tweet to the event. Only 3 event nodes are ever created.

**Step 3d: Create entity nodes and TARGETS/MENTIONS edges**

```python
targets_in_tweet = []
for ann in annotations:
    entity_name = ann["choice"]
    entity_id = f"{ENTITY_PREFIX}{entity_name}"

    if entity_id not in G:
        G.add_node(entity_id, node_type=NODE_TYPE_ENTITY, label=entity_name)

    is_target = ann["label"] == 1
    sentiment = ann.get("sentiment", "")

    if is_target:
        G.add_edge(tweet_id, entity_id, edge_type=EDGE_TARGETS,
                   sentiment=sentiment or "unknown", split=split)
        targets_in_tweet.append(entity_name)
    else:
        G.add_edge(tweet_id, entity_id, edge_type=EDGE_MENTIONS, split=split)
```

For each annotation on the tweet:
- `ann["choice"]` is the entity name (e.g., `"George Floyd"`, `"Donald Trump"`)
- `ann["label"]` is `1` (entity IS a target) or `0` (entity is NOT a target)
- If `label=1`, we create a TARGETS edge with sentiment info, and remember the entity for co-targeting
- If `label=0`, we create a MENTIONS edge (the entity was a candidate but not a target)

**The difference between TARGETS and MENTIONS is the core of the task.** A MENTIONS edge means the entity was presented as a candidate (it's plausibly related to the tweet) but human annotators decided the tweet does NOT target it. A TARGETS edge means annotators confirmed the tweet does target that entity, and the sentiment attribute captures whether the tweet is positive, negative, or neutral toward it.

**Step 3e: Create CO_TARGETED edges**

```python
for e1, e2 in combinations(sorted(targets_in_tweet), 2):
    G.add_edge(f"{ENTITY_PREFIX}{e1}", f"{ENTITY_PREFIX}{e2}",
               edge_type=EDGE_CO_TARGETED, tweet_url=tweet_url, split=split)
```

`combinations(sorted(targets_in_tweet), 2)` generates all unique pairs of entities that were **both** targeted in the same tweet. For example, if a tweet targets `["George Floyd", "Black people", "Breonna Taylor"]`, we get:
- (Black people, Breonna Taylor)
- (Black people, George Floyd)
- (Breonna Taylor, George Floyd)

These CO_TARGETED edges capture which entities tend to be discussed together. Sorting ensures consistent ordering. Each edge stores which tweet it came from and which split.

### save_graph()

Saves the graph in three formats:

```python
def save_graph(G, output_dir, prefix="tweet_target"):
```

1. **GraphML** (`tweet_target_graph.graphml`) — XML-based format readable by Gephi, Cytoscape, yEd, and any NetworkX program. All node/edge attributes are preserved as XML attributes. Human-readable but verbose.

2. **Pickle** (`tweet_target_graph.pkl`) — Python's binary serialization. Fastest to load back into Python, preserves the exact NetworkX object with all types intact. Not portable to other languages.

3. **JSON node-link** (`tweet_target_graph.json`) — Uses NetworkX's `json_graph.node_link_data()` format. Stores nodes as a list of objects and edges as a list of `{source, target, key, ...attributes}` objects. Good for web-based tools or any JSON-consuming application.

### print_graph_stats()

Computes and prints a comprehensive summary:

1. **Node counts by type** — How many tweet/entity/author/event/party nodes, plus average degree (number of connected edges) per type
2. **Edge counts by type** — How many AUTHORED/TARGETS/MENTIONS/ABOUT/CO_TARGETED/BELONGS_TO edges
3. **Sentiment distribution** — Among TARGETS edges, how many are positive/negative/neutral
4. **Author party distribution** — How many authors are Democrat/Republican/unknown
5. **Event distribution** — How many tweets belong to each of the 3 events
6. **Top 10 most connected entities** — Entities with the highest degree (most edges), indicating the most central/discussed entities
7. **Top 10 most prolific authors** — Authors who wrote the most tweets in the dataset

Returns a dict of all these stats (also saved to `graph_stats.json`).

### Main Block

```python
if __name__ == "__main__":
    G = build_graph(splits=["dev", "test"])
    stats = print_graph_stats(G)
    save_graph(G, config.OUTPUT_DIR / "graph")
    # Save stats JSON
```

When you run `python -m graph.graph_builder`, it:
1. Builds the graph for dev + test splits
2. Prints the full statistics table to stdout
3. Saves graph files to `output/graph/`
4. Saves the stats dict to `output/graph/graph_stats.json`

---

## graph/visualize.py

Generates interactive HTML visualizations using pyvis (a Python wrapper around the vis.js JavaScript library).

### Styling Constants

```python
NODE_STYLE = {
    "tweet":  {"color": "#4CAF50", "size": 8,  "shape": "dot"},     # green, small
    "entity": {"color": "#FF5722", "size": 15, "shape": "dot"},     # orange, medium
    "author": {"color": "#2196F3", "size": 12, "shape": "dot"},     # blue, medium
    "event":  {"color": "#9C27B0", "size": 30, "shape": "star"},    # purple, large star
    "party":  {"color": "#FF9800", "size": 25, "shape": "diamond"}, # orange, large diamond
}

EDGE_STYLE = {
    "TARGETS":     {"color": "#E53935", "width": 1.5},  # red
    "MENTIONS":    {"color": "#BDBDBD", "width": 0.5},  # grey, thin
    "AUTHORED":    {"color": "#1E88E5", "width": 1.0},  # blue
    "ABOUT":       {"color": "#8E24AA", "width": 1.0},  # purple
    "BELONGS_TO":  {"color": "#FB8C00", "width": 2.0},  # orange, thick
    "CO_TARGETED": {"color": "#FDD835", "width": 0.5},  # yellow, thin
}
```

Each node type gets a distinct color and shape so you can visually identify what you're looking at. Events are large stars (there are only 3), parties are diamonds (only 2), and tweets are tiny dots (628 of them).

**Helper functions:**
- `build_label(node_id, attrs)` — Creates the text shown next to each node. Tweets show `@author (date)`, entities/events show their name, authors show `@screen_name`.
- `build_title(node_id, attrs)` — Creates the HTML tooltip shown on hover. Lists all attributes of the node.

### visualize_full()

Renders all 1,150 nodes and 3,691 edges.

```python
net = Network(height="900px", width="100%", directed=True,
              bgcolor="#1a1a2e", font_color="white")
```

Creates a dark-themed full-width canvas. `directed=True` means edges have arrowheads.

```python
net.barnes_hut(gravity=-3000, central_gravity=0.3,
               spring_length=150, spring_strength=0.01, damping=0.09)
```

Configures the physics engine that lays out the graph. Barnes-Hut is an approximation algorithm for N-body simulation:
- `gravity=-3000` — Nodes repel each other (negative = repulsion). Higher magnitude = more spread out.
- `central_gravity=0.3` — Pulls everything toward the center so the graph doesn't fly apart.
- `spring_length=150` — Edges act like springs with a natural length of 150 pixels.
- `spring_strength=0.01` — How stiff those springs are (low = flexible layout).
- `damping=0.09` — How quickly the simulation settles down (higher = faster stabilization).

The function iterates over all nodes and edges, applying the styles, then saves to an HTML file.

### visualize_no_tweets() — The Condensed View

This is the more useful visualization. With 628 tweet nodes, the full graph is cluttered. The condensed view removes tweet nodes entirely and shows the **collapsed relationships**:

**How it works:**

1. **Collect author-to-entity relationships through tweets:**
   ```python
   for u, v, attrs in G.edges(data=True):
       if attrs.get("edge_type") == EDGE_TARGETS:
           tweet_attrs = G.nodes.get(u, {})
           screen_name = tweet_attrs.get("screen_name", "")
           if screen_name:
               author_id = f"author::{screen_name}"
               author_targets[author_id][v] += 1
   ```
   For every TARGETS edge (tweet → entity), it looks up which author wrote that tweet (from the tweet node's `screen_name` attribute) and records `author → entity` with a count. So if @LacyClayMO1 wrote 5 tweets that target "George Floyd", we get `author::LacyClayMO1 → entity::George Floyd` with count 5.

2. **Add only non-tweet nodes** — Skips all tweet nodes, scales entity node sizes based on their degree (more connections = bigger node).

3. **Add collapsed author → entity edges** — Color-coded by the dominant sentiment:
   - Green (#43A047) = mostly positive targeting
   - Red (#E53935) = mostly negative targeting
   - Yellow (#FDD835) = mostly neutral
   - Edge thickness scales with count (more tweets = thicker line)

4. **Add CO_TARGETED edges with frequency filtering** — Only shows entity pairs that were co-targeted at least twice (`count >= 2`), reducing clutter from one-off co-occurrences.

---

## The Resulting Graph

### Node Types and Their Attributes

| Type | Count | ID Format | Attributes |
|------|-------|-----------|------------|
| **tweet** | 628 | `tweet::https://www.twitter.com/.../1234` | `url`, `screen_name`, `date` (YYYY-MM-DD), `event`, `splits` (comma-separated) |
| **entity** | 318 | `entity::George Floyd` | `label` (entity name) |
| **author** | 199 | `author::LacyClayMO1` | `screen_name`, `party` ("Democrat"/"Republican"/"unknown") |
| **event** | 3 | `event::Death of George Floyd` | `label` (event name) |
| **party** | 2 | `party::Democrat` | `label` (party name) |

### Edge Types and Their Attributes

| Type | Count | Direction | Attributes | Meaning |
|------|-------|-----------|------------|---------|
| **TARGETS** | 972 | tweet → entity | `sentiment` (positive/negative/neutral), `split` | Human annotators confirmed this entity is a target of the tweet |
| **CO_TARGETED** | 866 | entity → entity | `tweet_url`, `split` | Both entities were targeted in the same tweet |
| **AUTHORED** | 628 | author → tweet | — | This politician wrote this tweet |
| **ABOUT** | 628 | tweet → event | `split` | This tweet is about this event (inferred from date) |
| **MENTIONS** | 548 | tweet → entity | `split` | Entity was a candidate but NOT a target (label=0) |
| **BELONGS_TO** | 49 | author → party | — | This politician belongs to this party |

### Graph Statistics

**Sentiment breakdown** (across 972 TARGETS edges):
- Positive: 421 (43.3%)
- Negative: 319 (32.8%)
- Neutral: 232 (23.9%)

**Party breakdown** (across 199 authors):
- Democrat: 39
- Republican: 10
- Unknown: 150 (party mapping only covers 57 politicians)

**Event breakdown** (across 628 tweets):
- Kavanaugh Supreme Court Nomination: 313 (mostly dev split)
- Death of George Floyd: 288 (mostly dev split)
- 2021 US Capitol Attack: 27 (test split)

**Top connected entities:** Brett Kavanaugh (206), Black people (186), Law enforcement in the US (165), Donald Trump (159), Republican Party (155)

---

## Output Files

All outputs go to `output/graph/`:

| File | Format | Size | Best For |
|------|--------|------|----------|
| `tweet_target_graph.graphml` | XML | ~1.5 MB | Gephi, Cytoscape, cross-tool interop |
| `tweet_target_graph.pkl` | Python binary | ~500 KB | Fast reload in Python scripts |
| `tweet_target_graph.json` | JSON node-link | ~2 MB | Web visualizations, JS tools |
| `graph_stats.json` | JSON | ~1 KB | Quick reference for graph metrics |
| `graph_full.html` | HTML + JS | ~2 MB | Interactive full graph in browser |
| `graph_condensed.html` | HTML + JS | ~500 KB | Interactive condensed view in browser |

---

## How to Run

```bash
# Build the graph (reads raw data, saves all formats + stats)
python -m graph.graph_builder

# Generate interactive HTML visualizations
python -m graph.visualize

# Open visualizations in browser
open output/graph/graph_full.html
open output/graph/graph_condensed.html
```

To build a graph with different splits (e.g., include train):

```python
from graph.graph_builder import build_graph, save_graph
G = build_graph(splits=["train", "dev", "test"])
save_graph(G, Path("output/graph_all"))
```

To load and query the graph in a script:

```python
import pickle
with open("output/graph/tweet_target_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Get all entities targeted by a specific author
author = "author::LacyClayMO1"
for _, tweet, d in G.out_edges(author, data=True):
    if d.get("edge_type") == "AUTHORED":
        for _, entity, ed in G.out_edges(tweet, data=True):
            if ed.get("edge_type") == "TARGETS":
                print(G.nodes[entity]["label"], ed.get("sentiment"))
```
