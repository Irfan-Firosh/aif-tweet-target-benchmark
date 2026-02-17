"""
Twitter bio context provider.

Loads author Twitter bios from twitter_user_bios.json.
"""

import json
import logging

import config

logger = logging.getLogger(__name__)


def load_twitter_bios() -> dict[str, str]:
    """Load twitter_user_bios.json → {screen_name: bio}."""
    if config.TWITTER_BIOS_FILE.exists():
        with open(config.TWITTER_BIOS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_author_bio(screen_name: str | None, bios: dict[str, str]) -> str | None:
    """Look up an author's Twitter bio by screen name."""
    if not screen_name:
        return None
    return bios.get(screen_name) or bios.get(f"@{screen_name}")
