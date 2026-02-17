"""Graph construction configuration: constants and event inference utilities."""

from datetime import datetime, timezone

# --- Twitter snowflake timestamp offset ---
TWITTER_EPOCH_OFFSET = 1288834974657  # ms

# --- Event inference by tweet date ---
EVENT_DATE_RANGES = {
    "Kavanaugh Supreme Court Nomination": {
        "start": datetime(2018, 9, 1, tzinfo=timezone.utc),
        "end": datetime(2018, 11, 1, tzinfo=timezone.utc),
    },
    "Death of George Floyd": {
        "start": datetime(2020, 5, 1, tzinfo=timezone.utc),
        "end": datetime(2020, 12, 1, tzinfo=timezone.utc),
    },
    "2021 US Capitol Attack": {
        "start": datetime(2021, 1, 1, tzinfo=timezone.utc),
        "end": datetime(2021, 4, 1, tzinfo=timezone.utc),
    },
}

# --- Node ID prefixes ---
TWEET_PREFIX = "tweet::"
ENTITY_PREFIX = "entity::"
AUTHOR_PREFIX = "author::"
EVENT_PREFIX = "event::"
PARTY_PREFIX = "party::"

# --- Node type labels ---
NODE_TYPE_TWEET = "tweet"
NODE_TYPE_ENTITY = "entity"
NODE_TYPE_AUTHOR = "author"
NODE_TYPE_EVENT = "event"
NODE_TYPE_PARTY = "party"

# --- Edge type labels ---
EDGE_AUTHORED = "AUTHORED"
EDGE_TARGETS = "TARGETS"
EDGE_MENTIONS = "MENTIONS"
EDGE_BELONGS_TO = "BELONGS_TO"
EDGE_ABOUT = "ABOUT"
EDGE_CO_TARGETED = "CO_TARGETED"


def status_id_to_datetime(status_id: str) -> datetime:
    """Convert a Twitter snowflake status ID to a UTC datetime."""
    ts_ms = (int(status_id) >> 22) + TWITTER_EPOCH_OFFSET
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def infer_event(tweet_url: str) -> str | None:
    """Infer the political event from the tweet URL's status ID timestamp."""
    parts = tweet_url.rstrip("/").split("/")
    status_id = parts[-1]
    try:
        dt = status_id_to_datetime(status_id)
    except (ValueError, IndexError, OverflowError):
        return None
    for event_name, date_range in EVENT_DATE_RANGES.items():
        if date_range["start"] <= dt <= date_range["end"]:
            return event_name
    return None
