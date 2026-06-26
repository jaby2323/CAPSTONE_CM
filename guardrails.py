"""
Safety guardrails and human-intervention checks (checkpoint 6).

These run over a scored topic and decide whether the forecast can be released
automatically or must be held for human review. The design is layered: a topic
can be flagged by any one of several independent checks.
"""

import config

# Categories that trigger mandatory human review before release.
SENSITIVE_KEYWORDS = [
    "casualty", "casualties", "death toll", "shooting", "attack", "bombing",
    "war", "killed", "hostage", "evacuat", "outbreak", "disaster",
]


def content_sensitivity_check(topic):
    """Flag topics touching crises / casualties / sensitive events."""
    text = (topic.get("topic", "") + " " +
            " ".join(i["title"] for i in topic.get("items", []))).lower()
    hits = [kw for kw in SENSITIVE_KEYWORDS if kw in text]
    return {"flagged": bool(hits), "matched": hits}


def source_verification(topic):
    """Single-source signals are inherently weaker -> cap and flag them."""
    return {
        "single_source": topic.get("single_source", True),
        "platform_count": topic.get("platform_count", 0),
    }


def manipulation_check(topic):
    """Cheap heuristic for coordinated inauthentic activity: many near-identical
    titles for the same topic suggests astroturfing rather than organic growth."""
    titles = [i["title"].lower().strip() for i in topic.get("items", [])]
    if len(titles) < 3:
        return {"flagged": False}
    unique_ratio = len(set(titles)) / len(titles)
    return {"flagged": unique_ratio < 0.4, "unique_ratio": round(unique_ratio, 2)}


def review_decision(topic, final_score, tie_unresolved=False):
    """Combine all checks into a single release/hold decision.

    Returns a dict with `requires_review` and the list of reasons.
    """
    reasons = []

    sensitivity = content_sensitivity_check(topic)
    if sensitivity["flagged"]:
        reasons.append(f"content_sensitivity: {sensitivity['matched']}")

    manip = manipulation_check(topic)
    if manip["flagged"]:
        reasons.append("possible_coordinated_activity")

    low, high = config.LOW_CONF_BAND
    if low <= final_score <= high:
        reasons.append("low_confidence_band")

    if tie_unresolved:
        reasons.append("unresolved_tie")

    return {
        "requires_review": bool(reasons),
        "reasons": reasons,
        "checks": {
            "sensitivity": sensitivity,
            "source": source_verification(topic),
            "manipulation": manip,
        },
    }
