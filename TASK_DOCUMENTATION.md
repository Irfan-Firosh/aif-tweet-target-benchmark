# Tweet Target Entity & Sentiment Detection

**Source**: "We Demand Justice!: Towards Social Context Grounding of Political Texts"
Pujari, Wu, & Goldwasser — EMNLP 2024
**Code/Data**: https://github.com/pujari-rajkumar/language-in-context

---

## Task Definition

Given a tweet `T`, its social context, and a candidate entity `E`, the model must predict:

1. **Target Identification** (binary) — Is `E` an intended target of `T`?
2. **Sentiment Identification** (4-class) — What is the sentiment toward `E`? One of `{positive, negative, neutral, non-target}`.

A "target entity" is any entity that would appear in the full, unambiguous interpretation of the tweet — even if it is **never explicitly mentioned** in the text. This forces models to reason about pragmatic intent using social context rather than surface-level string matching.

### Example

| Field | Value |
|-------|-------|
| **Tweet** | "As if we needed more evidence. #kavanaugh" |
| **Event** | Kavanaugh Supreme Court Nomination |
| **Author** | Earl Blumenauer (Democrat) |
| **Targets** | Brett Kavanaugh (negative), Julie Swetnick (positive), Christine Ford (positive), Deborah Ramirez (positive) |

None of the positive targets are named in the tweet text. Understanding that a Democrat politician referencing Kavanaugh with sarcasm implies support for his accusers requires social context.

---

## Data Schema

Each example is a tuple `⟨author, event, tweet, candidate_entity⟩` with a label.

### Input Fields

| Field | Description |
|-------|-------------|
| `tweet_url` | Twitter URL uniquely identifying the tweet (from Congress Tweets corpus) |
| `author` | The US politician who authored the tweet |
| `author_party` | `Democrat` or `Republican` |
| `event` | One of three political events (see below) |
| `candidate_entity` | An entity name to classify as target or non-target |

### Labels

| Label | Meaning |
|-------|---------|
| `label=1` | Entity **is** a target of the tweet |
| `label=0` | Entity **is not** a target |
| `sentiment` | Only present when `label=1`. One of: `positive`, `negative`, `neutral` |

### Dataset Split (`target_task_split_v7.json`)

The JSON is keyed by split → tweet_url → list of entity annotations:

```json
{
  "train": {
    "https://www.twitter.com/<screen_name>/statuses/<id>": [
      {"choice": "George Floyd", "sentiment": "positive", "label": 1},
      {"choice": "United States Senate", "label": 0}
    ]
  },
  "dev": { ... },
  "test": { ... }
}
```

Splits are divided **by event, author, and target** so the test set contains unseen combinations. The Capitol Riots event is reserved for testing.

| Split | Tweets | Targets | Non-Targets | Total Annotations |
|-------|--------|---------|-------------|-------------------|
| Train | 795 | 2,410 | 1,960 | 4,370 |
| Dev | 370 | 271 | 240 | 511 |
| Test | 441 | 701 | 308 | 1,009 |

---

## Events Covered

| Event | Split Usage |
|-------|-------------|
| Death of George Floyd | Train/Dev |
| Brett Kavanaugh Supreme Court Nomination | Train/Dev |
| 2021 US Capitol Attack | **Test only** |

Tweets were collected from the [Congress Tweets](https://github.com/alexlitel/congresstweets) corpus, filtered by hashtags, keywords, and date ranges. 1,779 media-containing tweets were selected to increase the likelihood that tweet text omits explicit entity mentions.

---

## Social Context Sources

The paper evaluates models at increasing levels of context. Our `data_collector.py` implements data loading for all three levels.

### 1. No Context
Only the tweet text, author name, event name, and entity name.

### 2. Twitter Bio Context
Author's Twitter biography text.
- **File**: `twitter_user_bios.json` — maps screen names to bio strings.

### 3. Wikipedia Context
Full Wikipedia page text for the author, event, and candidate entity.
- **Directory**: `wiki_pages_7.0/` — 812 pre-fetched Wikipedia pages stored as plain text files, named by entity/page title.
- **Fallback**: LangChain `WikipediaAPIWrapper` fetches pages not found locally.

### Supporting Files

| File | Contents |
|------|----------|
| `all_targets.json` | Raw list of all annotated target strings (362 unique) |
| `cleaned_targets.json` | Deduplicated/normalized target list |
| `target_map.json` | Maps raw mention variants → canonical entity name |
| `us_politician_names.json` | Screen name → real name mapping |
| `remaining_party_mapping.json` | Screen name → party affiliation |
| `background_texts/` | Topic background documents (guns, immigration, civil rights, etc.) |

---

## Annotation Process

- 6 in-house annotators + Amazon Mechanical Turk workers familiar with event context.
- Annotators marked target entities and sentiments per tweet, given a list of focal entities for each event plus the option to add new ones.
- Non-targets were chosen to be **event-relevant** (harder negatives).
- 3 annotators per tweet; majority agreement required.
- All AMT annotations were verified by in-house annotators.
- Inter-annotator agreement (Cohen's kappa): **0.47** for targets, **0.73** for sentiment.

---

## Baseline Results (from paper, Table 2)

### Target Identification (Binary)

| Model | Macro-F1 | Accuracy |
|-------|----------|----------|
| BERT-large (no context) | 68.83 | 70.56 |
| RoBERTa-base (no context) | 65.14 | 66.40 |
| BERT-large + Twitter Bio | 69.34 | 71.66 |
| BERT-large + Wikipedia | 60.33 | 61.05 |
| RoBERTa-base + Wikipedia | 68.62 | 70.27 |
| GPT-3 zero-shot | 69.77 | 73.78 |
| GPT-3 four-shot | 66.45 | 67.03 |
| RoBERTa-base + DCF Embs | **73.56** | **75.82** |
| BERT-large + DCF (fine-tuned) | 71.17 | 72.94 |

### Sentiment Identification (4-class)

| Model | Macro-F1 | Accuracy |
|-------|----------|----------|
| BERT-large (no context) | 58.95 | 58.37 |
| RoBERTa-base + DCF Embs | 62.90 | 63.03 |
| **BERT-large + DCF (fine-tuned)** | **65.34** | **65.31** |
| GPT-3 zero-shot | 54.18 | 56.80 |

### Key Findings

- **Context helps**: Models with social context consistently outperform no-context baselines.
- **Explicit > concatenated context**: DCF graph-based models beat simply appending Wikipedia text. Wikipedia-as-text actually hurt BERT-large on target identification (60.33 vs 68.83).
- **LLMs underperform fine-tuned models**: GPT-3 with textual context in the prompt performed worse than fine-tuned smaller models with structured context.
- **Large gap to human performance**: Humans achieve ~95% on comparable tasks.

---

## GPT Prompt Templates (from paper, Appendix A)

### Target Entity Detection

```
Event: <event>
Event background: <background-description>
Tweet: <tweet-text>
Author: <author-name>
Author Party: <party-affiliation>
Author background: <first two sentences of author-wiki-page>
Target Entity: <entity-name>
Entity background: <first two sentences of entity-wiki-page>

Task: Identify if the given entity is a target of the tweet.
A target entity is defined as an entity that would be present
in the full unambiguous explanation of the tweet.

Is the given entity a target entity of the tweet? Answer yes or no.
```

### Sentiment Detection

```
Event: <event>
Event background: <background-description>
Tweet: <tweet-text>
Author: <author-name>
Author Party: <party-affiliation>
Author background: <first two sentences of author-wiki-page>
Target Entity: <entity-name>
Entity background: <first two sentences of entity-wiki-page>

Task: Identify the sentiment of the tweet towards the given
target entity. Consider that the tweet is ambiguous and the
entity might be implied without being explicitly mentioned.

What is the sentiment of the tweet towards the target entity?
Answer with positive, negative, or neutral.
```

---

## Implementation Mapping

| Paper Concept | Our Implementation |
|---------------|-------------------|
| Dataset loading & splits | `data_collector.load_tweet_target_dataset()` |
| Target/sentiment annotations | `schemas.TargetAnnotation` |
| Tweet + metadata | `schemas.TweetTargetExample` |
| Wikipedia page context | `data_collector.get_wiki_context()` (local + API fallback) |
| Twitter bio context | `data_collector.load_twitter_bios()` |
| Author party/name resolution | `data_collector.load_party_mapping()`, `load_politician_names()` |
| Combined social context | `schemas.SocialContext` → `data_collector.build_social_context()` |
| Enriched example | `schemas.EnrichedTweetTargetExample` |
| Full pipeline | `data_collector.collect_all(enrich=True)` |
