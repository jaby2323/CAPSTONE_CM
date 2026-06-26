"""
Semantic vector store = the agent's long-term memory (checkpoints 2 & 3).

Each record is one past topic observation: its semantic summary, the source mix,
velocity at ingestion, a timestamp, and the eventual outcome label. Retrieval is
by cosine similarity over Gemini embeddings (semantic, not keyword) so the agent
can find historically similar topics even when the wording differs.

For the POC this is a simple JSON-backed store loaded into memory.
"""

import json
import os
from datetime import datetime, timezone

import config
from llm_clients import embed_text


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)


class VectorStore:
    def __init__(self, path=config.VECTOR_STORE_PATH):
        self.path = path
        self.records = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.records = json.load(f)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def add_record(self, topic_summary, metadata, outcome="pending"):
        """Write one topic observation back to long-term memory.

        Only the Forecast Agent should call this (tool-access guardrail).
        """
        record = {
            "summary": topic_summary,
            "embedding": embed_text(topic_summary),
            "metadata": metadata,
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.records.append(record)
        self._save()
        return record

    def query(self, topic_summary, top_k=config.RETRIEVAL_TOP_K):
        """Return the top-k most semantically similar historical records."""
        if not self.records:
            return []
        q = embed_text(topic_summary)
        scored = []
        for rec in self.records:
            sim = _cosine(q, rec["embedding"])
            # Return a lightweight view (drop the raw embedding) plus the score.
            scored.append({
                "summary": rec["summary"],
                "metadata": rec["metadata"],
                "outcome": rec["outcome"],
                "timestamp": rec["timestamp"],
                "similarity": round(sim, 3),
            })
        scored.sort(key=lambda r: r["similarity"], reverse=True)
        return scored[:top_k]

    def seed_demo_data(self):
        """Populate a few synthetic past records so retrieval has context to
        return on the very first run. Mirrors the checkpoint-3 example."""
        if self.records:
            return
        demo = [
            ("New AI regulation bill debated in legislature",
             {"platforms": ["reddit"], "velocity": 0.7}, "faded"),
            ("AI regulation discussion spikes in tech communities",
             {"platforms": ["reddit"], "velocity": 0.65}, "faded"),
            ("Major data breach at large retailer",
             {"platforms": ["reddit", "rss"], "velocity": 0.8}, "went_viral"),
            ("New consumer GPU launch generates buzz",
             {"platforms": ["reddit", "rss"], "velocity": 0.75}, "went_viral"),
            ("Routine quarterly earnings report",
             {"platforms": ["rss"], "velocity": 0.3}, "faded"),
        ]
        for summary, meta, outcome in demo:
            self.add_record(summary, meta, outcome)
        print(f"[vector_store] seeded {len(demo)} demo records.")
