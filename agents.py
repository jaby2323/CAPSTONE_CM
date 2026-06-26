"""
The four-agent system + the controller that sequences them (checkpoint 5).

    Collector -> Retrieval -> Critic (Tree-of-Thought) -> Forecast

Each agent has a single, clearly-scoped responsibility and passes a structured
payload to the next. The Controller is the lightweight orchestration layer
(checkpoint 5's "sequential with feedback" pattern). The Critic implements the
Tree-of-Thought beam search from checkpoint 4.
"""

import config
import tools
import guardrails
from llm_clients import call_claude, call_claude_json
from vector_store import VectorStore


# =============================================================================
# 1. COLLECTOR AGENT  -- live data retrieval + topic extraction
# =============================================================================
class CollectorAgent:
    """Fetches live data, validates it, extracts candidate topics, and computes
    deterministic momentum metrics. Scope is strictly input -- it does not score
    or interpret."""

    def run(self):
        # --- Act: fetch live data from multiple channels ---
        rss = tools.fetch_rss()
        hn = tools.fetch_hn()
        reddit = tools.fetch_reddit()  # best-effort; often blocked, degrades OK
        raw_items = rss + hn + reddit

        # --- Observe: is the data complete? (Reason: retry/flag if not) ---
        if len(raw_items) < 5:
            return {"ok": False, "reason": "insufficient_data",
                    "item_count": len(raw_items), "topics": []}

        # --- Reason + Act: cluster headlines into candidate topics (LLM tool) ---
        topics = self._extract_topics(raw_items)

        # --- Act: compute reproducible signal metrics for each topic ---
        topics = [tools.compute_signal_metrics(t) for t in topics]

        # Pre-filter by velocity so the ToT tree never gets too wide.
        topics.sort(key=lambda t: t["velocity"], reverse=True)
        topics = topics[:config.MAX_TOPICS_PER_CYCLE]

        return {"ok": True, "item_count": len(raw_items), "topics": topics}

    def _extract_topics(self, raw_items):
        """Use Claude to group raw headlines into emerging candidate topics."""
        # Number the items so the model can reference them by index.
        listing = "\n".join(
            f"{i}. [{it['platform']}|{it['source']}] {it['title']}"
            for i, it in enumerate(raw_items)
        )
        prompt = (
            "Below are live headlines/posts from RSS feeds and Reddit. Group them "
            "into at most 8 distinct emerging TOPICS. Merge items about the same "
            "story even if worded differently.\n\n"
            f"{listing}\n\n"
            'Return a JSON array of objects: '
            '[{"topic": "<short topic label>", "item_indices": [<ints>]}]'
        )
        clusters = call_claude_json(prompt, max_tokens=2000)

        topics = []
        for c in clusters:
            idxs = [i for i in c.get("item_indices", []) if 0 <= i < len(raw_items)]
            if not idxs:
                continue
            topics.append({
                "topic": c.get("topic", "unknown"),
                "items": [raw_items[i] for i in idxs],
            })
        return topics


# =============================================================================
# 2. RETRIEVAL AGENT  -- semantic RAG over long-term memory (checkpoint 3)
# =============================================================================
class RetrievalAgent:
    """Embeds each candidate topic and retrieves similar historical records from
    the vector store, packaging them alongside the live signal."""

    def __init__(self, store):
        self.store = store

    def run(self, topics, broaden=False):
        for t in topics:
            query = t["topic"]
            if broaden:  # feedback path: widen the query if analogies were poor
                query = t["topic"] + " " + " ".join(
                    i["title"] for i in t["items"][:2])
            t["retrieved"] = self.store.query(query)
        return topics


# =============================================================================
# 3. CRITIC AGENT  -- Tree-of-Thought + beam search evaluation (checkpoint 4)
# =============================================================================
class CriticAgent:
    """For each topic, generates optimistic/conservative/hedged interpretations
    and runs a beam search (width 2, depth 3) scoring each branch on the
    four-criteria rubric. Returns the winning branch per topic."""

    def evaluate(self, topic):
        # Depth 1: generate raw interpretations from velocity alone.
        branches = self._generate_branches(topic)
        for b in branches:
            b["score"] = self._score_branch(topic, b, use_history=False)
        branches = self._beam(topic, branches)

        # Depth 2: integrate retrieved historical context, re-score.
        for b in branches:
            b["score"] = self._score_branch(topic, b, use_history=True)
        branches = self._beam(topic, branches)

        # Depth 3: pick the surviving branch (cross-topic dedup handled by
        # the controller). Resolve ties via a reranking step.
        return self._select(topic, branches)

    # --- thought generator (LLM): three weightings of the signal ---
    def _generate_branches(self, topic):
        prompt = (
            f'Candidate topic: "{topic["topic"]}"\n'
            f'Signal: platforms={topic["platforms"]}, '
            f'velocity={topic["velocity"]}, source_count={topic["source_count"]}, '
            f'engagement={topic["total_engagement"]}.\n\n'
            "Produce THREE interpretations of whether this topic will gain "
            "mainstream traction in 24-48h: an optimistic, a conservative, and a "
            "hedged reading. For each give a stance and the key evidence.\n"
            'Return JSON: [{"stance":"optimistic|conservative|hedged",'
            '"reasoning":"...","primary_evidence":"...","discounted":"..."}]'
        )
        try:
            branches = call_claude_json(prompt, max_tokens=1500)
            return branches[:config.BRANCHING_FACTOR]
        except Exception:  # noqa: BLE001 - keep the POC running
            return [{"stance": s, "reasoning": "fallback", "primary_evidence": "",
                     "discounted": ""} for s in
                    ("optimistic", "conservative", "hedged")]

    # --- critic/evaluator: heuristic rules + LLM judgment -> composite score ---
    def _score_branch(self, topic, branch, use_history):
        # Heuristic component (deterministic, auditable).
        score = topic["velocity"]
        if topic["platform_count"] >= 2:
            score += 0.15  # cross-platform confirmation
        # Hard rule: single-platform topics are capped at 0.5.
        if topic["single_source"]:
            score = min(score, 0.5)

        # Historical base rate from retrieved analogues (depth-2 signal).
        if use_history and topic.get("retrieved"):
            viral = sum(1 for r in topic["retrieved"]
                        if r["outcome"] == "went_viral")
            base_rate = viral / len(topic["retrieved"])
            score = 0.6 * score + 0.4 * base_rate

        # Stance nudges the optimism of the reading.
        score += {"optimistic": 0.05, "hedged": 0.0,
                  "conservative": -0.05}.get(branch.get("stance"), 0.0)

        return max(0.0, min(score, 1.0))

    # --- beam search selection policy: keep top-`BEAM_WIDTH`, prune <0.3 ---
    def _beam(self, topic, branches):
        survivors = [b for b in branches if b["score"] >= config.PRUNE_BELOW]
        if not survivors:  # everything pruned -> keep best so we can flag it
            survivors = sorted(branches, key=lambda b: b["score"], reverse=True)
        survivors.sort(key=lambda b: b["score"], reverse=True)
        return survivors[:config.BEAM_WIDTH]

    # --- decision maker: final branch + tie handling ---
    def _select(self, topic, branches):
        branches.sort(key=lambda b: b["score"], reverse=True)
        best = branches[0]
        tie_unresolved = False

        if len(branches) >= 2 and abs(branches[0]["score"] - branches[1]["score"]) < 0.05:
            # Reranking step: ask the critic which evidence chain is more coherent.
            best, tie_unresolved = self._rerank(topic, branches[:2])

        low, high = config.LOW_CONF_BAND
        return {
            "topic": topic["topic"],
            "final_score": round(best["score"], 3),
            "winning_branch": best,
            "low_confidence": low <= best["score"] <= high,
            "tie_unresolved": tie_unresolved,
        }

    def _rerank(self, topic, tied):
        prompt = (
            f'Two scored interpretations of "{topic["topic"]}" are nearly tied.\n'
            f'A) {tied[0]}\nB) {tied[1]}\n\n'
            'Which has the more internally coherent evidence chain? '
            'Return JSON: {"winner":"A|B|tie"}'
        )
        try:
            ans = call_claude_json(prompt, max_tokens=400).get("winner", "tie")
        except Exception:  # noqa: BLE001
            ans = "tie"
        if ans == "B":
            return tied[1], False
        if ans == "tie":
            return tied[0], True  # genuinely unresolved -> human review
        return tied[0], False


# =============================================================================
# 4. FORECAST AGENT  -- structured output + write-back to memory
# =============================================================================
class ForecastAgent:
    """Turns a winning branch into a human-readable briefing with an evidence
    chain (output constraint), applies guardrails, and -- as the ONLY writer --
    records the outcome back to the vector store."""

    def __init__(self, store):
        self.store = store

    def run(self, topic, evaluation):
        review = guardrails.review_decision(
            topic, evaluation["final_score"],
            tie_unresolved=evaluation["tie_unresolved"])

        briefing = self._write_briefing(topic, evaluation, review)

        # Write the observation back to long-term memory (outcome unknown yet).
        self.store.add_record(
            topic_summary=topic["topic"],
            metadata={
                "platforms": topic["platforms"],
                "velocity": topic["velocity"],
                "final_score": evaluation["final_score"],
            },
            outcome="pending",
        )

        return {
            "topic": topic["topic"],
            "confidence": evaluation["final_score"],
            "status": "HELD_FOR_REVIEW" if review["requires_review"] else "RELEASED",
            "review": review,
            "briefing": briefing,
            "evidence_chain": {  # output constraint: never an unexplained assertion
                "live_signal": {
                    "platforms": topic["platforms"],
                    "velocity": topic["velocity"],
                    "engagement": topic["total_engagement"],
                },
                "retrieved_analogues": topic.get("retrieved", []),
                "winning_interpretation": evaluation["winning_branch"],
            },
        }

    def _write_briefing(self, topic, evaluation, review):
        prompt = (
            "Write a 2-3 sentence forecast briefing for a knowledge worker.\n"
            f'Topic: "{topic["topic"]}"\n'
            f'Confidence: {evaluation["final_score"]} '
            f'(low_confidence={evaluation["low_confidence"]})\n'
            f'Signal: platforms={topic["platforms"]}, velocity={topic["velocity"]}\n'
            f'Winning interpretation: {evaluation["winning_branch"].get("reasoning","")}\n'
            f'Historical analogues: '
            f'{[(r["summary"], r["outcome"]) for r in topic.get("retrieved", [])]}\n'
            f'Review flags: {review["reasons"]}\n\n'
            "State the forecast, the main driver, and any uncertainty. Be concise."
        )
        try:
            return call_claude(prompt, max_tokens=400)
        except Exception as e:  # noqa: BLE001
            return f"(briefing unavailable: {e})"


# =============================================================================
# CONTROLLER  -- sequences the agents with the Critic->Retrieval feedback hop
# =============================================================================
class ForecastController:
    def __init__(self, store=None):
        self.store = store or VectorStore()
        self.collector = CollectorAgent()
        self.retrieval = RetrievalAgent(self.store)
        self.critic = CriticAgent()
        self.forecast = ForecastAgent(self.store)

    def run_cycle(self, verbose=True):
        log = print if verbose else (lambda *a, **k: None)

        # --- Collector ---
        log("[1/4] Collector: fetching live data...")
        collected = self.collector.run()
        if not collected["ok"]:
            log(f"  data not OK: {collected['reason']}")
            return {"ok": False, "detail": collected}
        topics = collected["topics"]
        log(f"  {collected['item_count']} items -> {len(topics)} candidate topics")

        # --- Retrieval (RAG) ---
        log("[2/4] Retrieval: querying long-term memory...")
        topics = self.retrieval.run(topics)

        # --- Critic (Tree-of-Thought) with one feedback hop ---
        log("[3/4] Critic: Tree-of-Thought scoring...")
        evaluations = []
        for t in topics:
            ev = self.critic.evaluate(t)
            # Feedback: if retrieved analogues look weak, ask Retrieval to broaden.
            avg_sim = (sum(r["similarity"] for r in t.get("retrieved", [])) /
                       max(len(t.get("retrieved", [])), 1))
            if avg_sim < 0.3 and t.get("retrieved"):
                log(f"  weak analogues for '{t['topic']}' -> broadening retrieval")
                self.retrieval.run([t], broaden=True)
                ev = self.critic.evaluate(t)
            evaluations.append(ev)

        # --- Forecast + write-back ---
        log("[4/4] Forecast: generating briefings...")
        results = [self.forecast.run(t, ev)
                   for t, ev in zip(topics, evaluations)]
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return {"ok": True, "results": results}
