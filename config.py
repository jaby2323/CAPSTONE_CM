"""
Central configuration for the Trending Topic & Viral Signal Forecaster.

Holds model IDs, data sources, and the tuning constants referenced across the
checkpoints (beam width, ToT depth, pruning thresholds, retrieval top-k, etc.).
Keeping these in one place makes the agent behaviour easy to inspect and tweak.
"""

import os

# --- LLM configuration -------------------------------------------------------
# Claude is the reasoning model (Critic + Forecast agents).
# Gemini is used for semantic embeddings (the RAG layer).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

CLAUDE_MODEL = "claude-opus-4-8"          # reasoning model (checkpoint 2/4/5)
GEMINI_EMBED_MODEL = "gemini-embedding-001"  # semantic retrieval (checkpoint 3)

# --- Data sources (checkpoint 1 & 5: Collector Agent) ------------------------
# RSS feeds from major outlets. Easy to extend.
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "http://rss.cnn.com/rss/edition.rss",
    "https://www.theverge.com/rss/index.xml",
    "https://techcrunch.com/feed/",
]

# Public subreddits to monitor (Reddit's public .json endpoint, no auth needed).
SUBREDDITS = ["technology", "worldnews", "news", "science"]

# --- Reasoning / search parameters (checkpoint 4: Tree-of-Thought) -----------
BEAM_WIDTH = 2          # branches kept per topic at each depth
TOT_DEPTH = 3           # depth limit (velocity -> +history -> +cross-topic)
BRANCHING_FACTOR = 3    # optimistic / conservative / hedged interpretations
MAX_TOPICS_PER_CYCLE = 8  # pre-filter cap so the tree never gets too wide

# Pruning thresholds (checkpoint 4 rubric)
PRUNE_BELOW = 0.3       # eliminate immediately
LOW_CONF_BAND = (0.3, 0.5)  # low-confidence / human-review band

# --- Retrieval parameters (checkpoint 3: RAG) --------------------------------
RETRIEVAL_TOP_K = 4     # 3-5 similar historical records per query
STALE_DAYS = 90         # discount retrieved analogues older than this

# --- Persistence -------------------------------------------------------------
VECTOR_STORE_PATH = "vector_store.json"  # the agent's accumulated history
