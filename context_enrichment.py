"""
Social context enrichment pipeline.

Combines Wikipedia context + Twitter bios to build
full SocialContext for each TweetTargetExample.

Usage:
    from context_enrichment import enrich_dataset
    enriched = enrich_dataset(raw_examples)
"""

import logging
from pathlib import Path

from langchain_community.utilities import WikipediaAPIWrapper

from bio_context import get_author_bio, load_twitter_bios
from data_collector import extract_screen_name, load_tweet_target_dataset, save_dataset
from schemas import EnrichedTweetTargetExample, SocialContext, TweetTargetExample
from wiki_context import create_wiki_api, get_wiki_context, load_local_wiki_page
import config

logger = logging.getLogger(__name__)


def load_background_text(directory: Path, topic: str) -> str | None:
    """Load a background .txt file for a topic."""
    path = directory / f"{topic}_background.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace").strip()
    return None


def build_social_context(
    example: TweetTargetExample,
    bios: dict[str, str],
    wiki_api: WikipediaAPIWrapper,
) -> SocialContext:
    """Build social context for a tweet target example."""
    screen_name = extract_screen_name(example.tweet_url)
    author_bio = get_author_bio(screen_name, bios)

    # Author Wikipedia page
    author_ctx = None
    if screen_name:
        author_ctx = get_wiki_context(f"@{screen_name}", wiki_api)
        if not author_ctx and example.author:
            author_ctx = get_wiki_context(example.author, wiki_api)

    # Event context
    event_ctx = None
    if example.event:
        event_ctx = get_wiki_context(example.event, wiki_api)

    # Entity contexts (for targets only)
    entity_contexts = []
    for t in example.targets:
        if t.is_target:
            ctx = get_wiki_context(t.entity, wiki_api)
            if ctx:
                entity_contexts.append(ctx)

    return SocialContext(
        event_context=event_ctx,
        author_context=author_ctx,
        entity_contexts=entity_contexts,
        author_bio=author_bio,
    )


def enrich_dataset(
    examples: list[TweetTargetExample],
) -> list[EnrichedTweetTargetExample]:
    """
    Enrich a list of TweetTargetExamples with full social context.

    Fetches Wikipedia pages (local first, API fallback) and Twitter bios.
    """
    wiki_api = create_wiki_api()
    bios = load_twitter_bios()

    logger.info("Enriching tweet target examples with social context...")
    enriched = []
    for ex in examples:
        ctx = build_social_context(ex, bios, wiki_api)
        enriched.append(EnrichedTweetTargetExample(example=ex, context=ctx))
    logger.info(f"Enriched {len(enriched)} tweet target examples")

    return enriched


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw = load_tweet_target_dataset()
    enriched = enrich_dataset(raw)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_dataset(enriched, config.OUTPUT_DIR / "tweet_target_enriched.json")

    # Show a sample
    print("\n--- Local Wikipedia Pages Available ---")
    wiki_pages = [p for p in config.WIKI_PAGES_DIR.glob("*") if not p.name.endswith(".parse")]
    print(f"  {len(wiki_pages)} pages in wiki_pages_7.0/")

    for entity in ["George Floyd", "Donald Trump", "Brett Kavanaugh", "Joe Biden"]:
        ctx = load_local_wiki_page(entity)
        if ctx:
            preview = ctx.summary[:120].replace("\n", " ")
            print(f"  [local] {entity}: {preview}...")
        else:
            print(f"  [missing] {entity}: not found locally")
