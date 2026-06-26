"""
External data-retrieval tools and the deterministic signal-scoring functions.

These are the non-LLM "tools" from checkpoint 2:
  * fetch_rss / fetch_reddit  -> live data retrieval (grounds the agent in reality)
  * compute_signal_metrics    -> structured scoring (velocity, cross-platform)

The scoring math lives here, NOT in a prompt, so the numbers are reproducible.
"""

import time

import requests
import feedparser

import config

# A browser-like UA; Reddit blocks generic/default request agents.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}


# --- Live data retrieval -----------------------------------------------------
def fetch_rss(feeds=None, per_feed=15):
    """Fetch recent headlines from the configured RSS feeds."""
    feeds = feeds or config.RSS_FEEDS
    items = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            source = parsed.feed.get("title", url)
            for entry in parsed.entries[:per_feed]:
                items.append({
                    "title": entry.get("title", "").strip(),
                    "platform": "rss",
                    "source": source,
                    "engagement": 0,  # RSS gives no engagement signal
                })
        except Exception as e:  # noqa: BLE001
            print(f"[rss] failed to fetch {url}: {e}")
    return items


def fetch_hn(limit=25):
    """Fetch top Hacker News stories (a news-aggregator channel, checkpoint 1).

    HN's public Firebase API is unauthenticated and reliable, giving us a second
    platform so cross-platform confirmation can actually trigger."""
    items = []
    try:
        top = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            headers=HEADERS, timeout=15).json()
        for story_id in top[:limit]:
            d = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                headers=HEADERS, timeout=10).json() or {}
            if not d.get("title"):
                continue
            items.append({
                "title": d.get("title", "").strip(),
                "platform": "hackernews",
                "source": "Hacker News",
                "engagement": d.get("score", 0) + d.get("descendants", 0),
            })
    except Exception as e:  # noqa: BLE001
        print(f"[hn] failed to fetch Hacker News: {e}")
    return items


def fetch_reddit(subreddits=None, limit=20):
    """Fetch hot posts from public subreddits via Reddit's .json endpoint."""
    subreddits = subreddits or config.SUBREDDITS
    items = []
    for sub in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit={limit}"
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            for child in r.json().get("data", {}).get("children", []):
                d = child["data"]
                items.append({
                    "title": d.get("title", "").strip(),
                    "platform": "reddit",
                    "source": f"r/{sub}",
                    # comment + upvote velocity proxy
                    "engagement": d.get("num_comments", 0) + d.get("score", 0),
                })
            time.sleep(0.5)  # be polite to the public endpoint
        except Exception as e:  # noqa: BLE001
            print(f"[reddit] failed to fetch r/{sub}: {e}")
    return items


# --- Deterministic signal scoring -------------------------------------------
def compute_signal_metrics(topic):
    """Compute reproducible momentum metrics for one candidate topic.

    `topic` is a dict produced by the Collector with at least:
        items: list of source items belonging to this topic
    Returns the topic enriched with velocity / cross-platform fields.
    """
    items = topic.get("items", [])
    platforms = sorted({i["platform"] for i in items})
    sources = sorted({i["source"] for i in items})
    total_engagement = sum(i.get("engagement", 0) for i in items)

    # Simple, explainable velocity proxy: how much discussion volume this topic
    # carries, normalised to a 0-1 range for the scoring rubric.
    raw_velocity = len(items) + total_engagement / 100.0
    velocity = min(raw_velocity / 20.0, 1.0)

    topic.update({
        "platforms": platforms,
        "source_count": len(sources),
        "platform_count": len(platforms),
        "item_count": len(items),
        "total_engagement": total_engagement,
        "velocity": round(velocity, 3),
        "single_source": len(platforms) < 2,  # guardrail flag
    })
    return topic
