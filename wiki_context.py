"""
Wikipedia context provider.

Loads Wikipedia social context from:
  - Local wiki pages (data/tweet_target_data/wiki_pages_7.0/)
  - LangChain Wikipedia API as fallback for missing entities
"""

import logging
import urllib.parse

from langchain_community.utilities import WikipediaAPIWrapper

import config
from schemas import WikiContext

logger = logging.getLogger(__name__)


def load_local_wiki_page(title: str) -> WikiContext | None:
    """Load a Wikipedia page from the local wiki_pages_7.0 directory."""
    candidates = [
        config.WIKI_PAGES_DIR / title,
        config.WIKI_PAGES_DIR / urllib.parse.quote(title, safe=""),
    ]
    for path in candidates:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            return WikiContext(title=title, summary=text.strip(), source="local")
    return None


def fetch_wiki_page(title: str, wiki_api: WikipediaAPIWrapper) -> WikiContext | None:
    """Fetch a Wikipedia page via LangChain WikipediaAPIWrapper."""
    try:
        results = wiki_api.load(title)
        if results:
            page = results[0]
            return WikiContext(
                title=page.metadata.get("title", title),
                summary=page.page_content,
                source="api",
            )
    except Exception as e:
        logger.debug(f"Wikipedia API lookup failed for '{title}': {e}")
    return None


def get_wiki_context(title: str, wiki_api: WikipediaAPIWrapper) -> WikiContext | None:
    """Try local first, fall back to API."""
    ctx = load_local_wiki_page(title)
    if ctx:
        return ctx
    return fetch_wiki_page(title, wiki_api)


def create_wiki_api() -> WikipediaAPIWrapper:
    """Create a configured WikipediaAPIWrapper instance."""
    return WikipediaAPIWrapper(
        lang=config.WIKI_LANG,
        top_k_results=1,
        doc_content_chars_max=4000,
    )
