"""
Project configuration: paths, API settings, and constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# Tweet Target Entity & Sentiment task
TWEET_TARGET_DIR = DATA_DIR / "tweet_target_data"
TARGET_SPLIT_FILE = TWEET_TARGET_DIR / "data" / "target_task_split_v7.json"
ALL_TARGETS_FILE = TWEET_TARGET_DIR / "all_targets.json"
CLEANED_TARGETS_FILE = TWEET_TARGET_DIR / "cleaned_targets.json"
TARGET_MAP_FILE = TWEET_TARGET_DIR / "target_map.json"
WIKI_PAGES_DIR = TWEET_TARGET_DIR / "wiki_pages_7.0"
TWITTER_BIOS_FILE = TWEET_TARGET_DIR / "twitter_user_bios.json"
POLITICIAN_NAMES_FILE = TWEET_TARGET_DIR / "us_politician_names.json"
PARTY_MAPPING_FILE = TWEET_TARGET_DIR / "remaining_party_mapping.json"
TWEET_BACKGROUND_DIR = TWEET_TARGET_DIR / "background_texts"

# Output
OUTPUT_DIR = PROJECT_ROOT / "output"

# --- API settings ---
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")
GENAI_BASE_URL = "https://genai.rcac.purdue.edu/api"
DEFAULT_MODEL = "llama3.1:latest"

# --- Wikipedia ---
WIKI_LANG = "en"
WIKI_SENTENCES = 10  # Number of sentences to fetch per Wikipedia summary
