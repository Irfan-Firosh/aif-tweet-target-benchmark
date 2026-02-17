"""
Data schemas for Social Context Grounding benchmark.

Based on: "We Demand Justice!: Towards Social Context Grounding of Political Texts"
(Pujari et al., EMNLP 2024)

Task: Tweet Target Entity & Sentiment Detection
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


# --- Enums ---

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Party(str, Enum):
    DEMOCRAT = "Democrat"
    REPUBLICAN = "Republican"


# --- Tweet Target Entity & Sentiment ---

class TargetAnnotation(BaseModel):
    """A single target entity annotation for a tweet."""
    entity: str = Field(description="The target entity name (e.g., 'George Floyd', 'Donald Trump')")
    is_target: bool = Field(description="Whether this entity is a target of the tweet")
    sentiment: Optional[Sentiment] = Field(
        default=None,
        description="Sentiment towards the entity (only if is_target=True)"
    )


class TweetTargetExample(BaseModel):
    """A single example in the Tweet Target Entity & Sentiment task."""
    tweet_url: str = Field(description="Twitter URL identifying the tweet")
    tweet_text: Optional[str] = Field(default=None, description="The tweet text content")
    author: Optional[str] = Field(default=None, description="Author screen name")
    author_party: Optional[Party] = Field(default=None, description="Author's party affiliation")
    event: Optional[str] = Field(default=None, description="Associated political event")
    targets: list[TargetAnnotation] = Field(default_factory=list, description="Target annotations")
    split: str = Field(description="Dataset split: train/dev/test")


# --- Wikipedia Context ---

class WikiContext(BaseModel):
    """Wikipedia-sourced social context for an entity, event, or author."""
    title: str = Field(description="Wikipedia page title")
    summary: str = Field(description="Extracted summary or full page text")
    source: str = Field(
        default="local",
        description="'local' if from existing wiki_pages_7.0, 'api' if freshly fetched"
    )


class SocialContext(BaseModel):
    """Aggregated social context for a single data example."""
    event_context: Optional[WikiContext] = Field(default=None, description="Wikipedia context for the event")
    author_context: Optional[WikiContext] = Field(default=None, description="Wikipedia context for the author/party")
    entity_contexts: list[WikiContext] = Field(
        default_factory=list,
        description="Wikipedia context for target entities"
    )
    author_bio: Optional[str] = Field(default=None, description="Twitter bio of the author")
    background_text: Optional[str] = Field(default=None, description="Topic background text")


# --- Enriched example (data + context) ---

class EnrichedTweetTargetExample(BaseModel):
    """Tweet target example enriched with social context."""
    example: TweetTargetExample
    context: SocialContext
