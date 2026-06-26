"""
Thin wrappers around the two external models the agent depends on.

  * call_claude(...)      -> text / JSON reasoning   (Anthropic SDK)
  * embed_text(...)       -> semantic embedding      (Gemini REST API)

These are the "external tools" of checkpoint 2: live LLM reasoning and the
embedding function that powers semantic retrieval. Both degrade gracefully so
the proof-of-concept always runs even if a network call fails.
"""

import json
import hashlib
import requests

import anthropic

import config

# Single shared Claude client (reads ANTHROPIC_API_KEY from the environment).
_claude = anthropic.Anthropic() if config.ANTHROPIC_API_KEY else None


def call_claude(prompt, system=None, max_tokens=2000):
    """Send one prompt to Claude and return the text response."""
    if _claude is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
    resp = _claude.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system or "You are a precise analytical assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def call_claude_json(prompt, system=None, max_tokens=2000):
    """Call Claude expecting a JSON reply, and parse it robustly."""
    instruction = "\n\nRespond with ONLY valid JSON. No prose, no markdown fences."
    raw = call_claude(prompt + instruction, system=system, max_tokens=max_tokens)
    return _extract_json(raw)


def _extract_json(text):
    """Pull the first JSON object/array out of a model reply."""
    text = text.strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to grabbing the outermost braces/brackets.
        for open_c, close_c in (("{", "}"), ("[", "]")):
            start, end = text.find(open_c), text.rfind(close_c)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"Could not parse JSON from model output:\n{text[:500]}")


def embed_text(text):
    """Return a semantic embedding vector for `text` using Gemini.

    Falls back to a deterministic hash-based pseudo-embedding if the API key is
    missing or the call fails, so the RAG layer is always usable in the POC.
    """
    if config.GEMINI_API_KEY:
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{config.GEMINI_EMBED_MODEL}:embedContent?key={config.GEMINI_API_KEY}"
            )
            body = {
                "model": f"models/{config.GEMINI_EMBED_MODEL}",
                "content": {"parts": [{"text": text}]},
            }
            r = requests.post(url, json=body, timeout=20)
            r.raise_for_status()
            return r.json()["embedding"]["values"]
        except Exception as e:  # noqa: BLE001 - POC: any failure -> fallback
            print(f"[embed] Gemini call failed ({e}); using fallback embedding.")
    return _hash_embedding(text)


def _hash_embedding(text, dim=256):
    """Cheap deterministic fallback embedding (bag-of-hashed-words)."""
    vec = [0.0] * dim
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]
