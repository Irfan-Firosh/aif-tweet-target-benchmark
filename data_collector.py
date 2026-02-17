"""
Phase 1: Data Collection — No Context baseline.

Loads the Tweet Target Entity & Sentiment Detection dataset
from Pujari et al. (EMNLP 2024). Only raw data, no context enrichment.

Usage:
    python data_collector.py
"""

import json
import logging
from pathlib import Path

import config
from schemas import (
    Party,
    Sentiment,
    TargetAnnotation,
    TweetTargetExample,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_party_mapping() -> dict[str, str]:
    """Load remaining_party_mapping.json → {screen_name: party}."""
    if config.PARTY_MAPPING_FILE.exists():
        return load_json(config.PARTY_MAPPING_FILE)
    return {}


def load_politician_names() -> dict[str, str]:
    """Load us_politician_names.json → {screen_name: real_name}."""
    if config.POLITICIAN_NAMES_FILE.exists():
        return load_json(config.POLITICIAN_NAMES_FILE)
    return {}


def extract_screen_name(tweet_url: str) -> str | None:
    """Extract screen name from a tweet URL like https://www.twitter.com/AlLawsonJr/statuses/..."""
    parts = tweet_url.rstrip("/").split("/")
    try:
        idx = parts.index("twitter.com")
        return parts[idx + 1] if idx + 1 < len(parts) else None
    except ValueError:
        return None


def load_tweet_target_dataset() -> list[TweetTargetExample]:
    """
    Load the Tweet Target Entity & Sentiment dataset from target_task_split_v7.json.
    Returns structured TweetTargetExample objects for train/dev/test splits.
    """
    raw = load_json(config.TARGET_SPLIT_FILE)
    party_map = load_party_mapping()
    name_map = load_politician_names()

    examples = []
    for split_name, tweets in raw.items():
        for tweet_url, annotations in tweets.items():
            screen_name = extract_screen_name(tweet_url)

            targets = []
            for ann in annotations:
                is_target = ann["label"] == 1
                sentiment = None
                if is_target and "sentiment" in ann:
                    sentiment = Sentiment(ann["sentiment"])
                targets.append(TargetAnnotation(
                    entity=ann["choice"],
                    is_target=is_target,
                    sentiment=sentiment,
                ))

            author_party = None
            if screen_name:
                party_str = party_map.get(f"@{screen_name}") or party_map.get(screen_name)
                if party_str in ("D", "Democrat"):
                    author_party = Party.DEMOCRAT
                elif party_str in ("R", "Republican"):
                    author_party = Party.REPUBLICAN

            examples.append(TweetTargetExample(
                tweet_url=tweet_url,
                author=name_map.get(screen_name, screen_name) if screen_name else None,
                author_party=author_party,
                targets=targets,
                split=split_name,
            ))

    logger.info(
        f"Loaded tweet target dataset: "
        f"{sum(1 for e in examples if e.split == 'train')} train, "
        f"{sum(1 for e in examples if e.split == 'dev')} dev, "
        f"{sum(1 for e in examples if e.split == 'test')} test"
    )
    return examples


def save_dataset(examples: list, output_path: Path) -> None:
    """Save a list of Pydantic model instances to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [ex.model_dump() for ex in examples]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} examples to {output_path}")


if __name__ == "__main__":
    logger.info("=== Phase 1: Data Collection (No Context) ===")

    tweet_examples = load_tweet_target_dataset()

    print("\n--- Tweet Target Entity & Sentiment Dataset ---")
    for split in ("train", "dev", "test"):
        split_ex = [e for e in tweet_examples if e.split == split]
        total_targets = sum(
            sum(1 for t in e.targets if t.is_target) for e in split_ex
        )
        total_non_targets = sum(
            sum(1 for t in e.targets if not t.is_target) for e in split_ex
        )
        print(f"  {split:>5}: {len(split_ex)} tweets, "
              f"{total_targets} targets, {total_non_targets} non-targets")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_dataset(tweet_examples, config.OUTPUT_DIR / "tweet_target_raw.json")
    print(f"\nRaw dataset saved to {config.OUTPUT_DIR}/")
