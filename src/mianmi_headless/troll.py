"""Troll toll — simplified for the headless harness.

Mirrors the full mianmi project's pre-tool-call pattern check, minus
the DB-backed catch list. The headless harness has no Postgres, so
the seed list is the only source of catches. That's fine for the
harbor benchmark case — the user wants to catch the obvious "stale
training data" failures, not build an overnight gardener pipeline.

The toll fires on:
  - Stale model ids: claude-sonnet-4, minimax-m2, kimi-k2, anthropic/claude
  - Stale phrasing: "the latest model", "the most recent public",
    "as of my training"
  - Heredoc in shell commands (Codex trip-on pattern)
  - Anthropic as a default provider

When the toll fires, the tool call is canceled and the agent sees
the lesson + 4 resolution paths: search, rewrite, defend, escalate.
The agent can pay the toll with a one-sentence defense
(via ``toll_pay_toll`` in the agent's tool list, but for the
headless harness we just return the lesson as the tool result
and let the agent's reasoning handle the resolution).

This module deliberately has no DB, no async, no model. The point
is to be the cheapest possible insurance against the most common
training-data drift failure.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# The seed list — the only catches the headless harness ships with.
# These are the patterns the user has personally tripped on, or the
# ones the agent is most likely to add once the full mianmi project
# is in place.
# --------------------------------------------------------------------------- #

SEED_CATCHES: list[dict[str, str]] = [
    {
        "kind": "model_id_stale",
        "pattern": "claude-sonnet-4",
        "lesson": (
            "You just used 'claude-sonnet-4' or similar. That's a stale "
            "model id from your training data. The user explicitly does "
            "NOT want Claude/Anthropic for any work. The current front "
            "model is gpt-5.5 (1.05M context, OpenAI direct or via "
            "OpenRouter). Before suggesting a model id, run "
            "scrape_openrouter_models to see what's actually live today."
        ),
    },
    {
        "kind": "phrasing_stale",
        "pattern": "the latest model",
        "lesson": (
            "You said 'the latest model' or 'the most recent public model' "
            "or similar. You have no way to know what's 'latest' — your "
            "training data is months old. Either run a web search to "
            "verify, or name the specific model you mean."
        ),
    },
    {
        "kind": "phrasing_stale",
        "pattern": "the most recent public",
        "lesson": (
            "You said 'the most recent public' or similar phrasing that "
            "implies you know what's current. You don't. Run a search "
            "or name the specific item."
        ),
    },
    {
        "kind": "phrasing_stale",
        "pattern": "as of my (last )?training",
        "lesson": (
            "You said 'as of my training' or 'as of my last training'. "
            "Your training data is months stale. Run a web search to "
            "verify current information."
        ),
    },
    {
        "kind": "model_id_stale",
        "pattern": "minimax-m2",
        "lesson": (
            "You said 'minimax-m2' (or 'minimax-m2.5' / 'minimax-m2.7' / "
            "similar). That's the previous-generation Minimax model. The "
            "current model is Minimax-M3 (released ~2026-06-03), 1M "
            "context — used as the traveling gardener in this harness."
        ),
    },
    {
        "kind": "model_id_stale",
        "pattern": "kimi-k2",
        "lesson": (
            "You said 'kimi-k2' (or 'kimi-k2.5' / 'kimi-k2.7' / similar). "
            "The current Kimi is K2.6, but it's NOT a valid gardener — "
            "it tops out at 128K context. The headless harness uses M3 "
            "(1M context) as the gardener."
        ),
    },
    {
        "kind": "tool_misuse",
        "pattern": r"<<\s*EOF|cat\s*<<\s*EOF",
        "lesson": (
            "You're writing a heredoc. Codex has tripped on chained "
            "heredoc + pipe syntax multiple times. The correct pattern: "
            "pass the heredoc body as a single 'input' arg, OR use "
            "`bash -c '...' <<< \"$var\"` for single-line, OR if you "
            "must use <<EOF in a command string, make sure the closing "
            "EOF is on a line by itself with no leading whitespace. "
            "If you really need the existing syntax, defend it in one "
            "sentence."
        ),
    },
    {
        "kind": "provider_default",
        "pattern": r"anthropic/claude|openrouter/anthropic",
        "lesson": (
            "You picked an Anthropic model as a default. The user has "
            "explicitly said: do NOT use Claude/Anthropic for production "
            "work. Default is gpt-5.5 (OpenAI direct). Anthropic is "
            "opt-in only."
        ),
    },
]


# Paperbench-specific seed list. These are the patterns the agent
# is MOST likely to hit on a paperbench task. Activated only when
# the user sets ``MIANMI_HEADLESS_PAPERBENCH=1`` (no auto-detect
# from the verifier's rubric file, since that's a cheating surface).

PAPERBENCH_CATCHES: list[dict[str, str]] = [
    {
        "kind": "package_upgrade",
        "pattern": r"upgrade\s+(torch|transformers|accelerate|datasets)|pip\s+install\s+--upgrade",
        "lesson": (
            "You're suggesting a package upgrade. STOP. The paperbench "
            "harness pins specific versions in the environment; "
            "upgrading risks breaking compatibility with the rubric's "
            "runtime checks (L004: 'pins and reports torch / "
            "transformers / accelerate / datasets versions'). Use the "
            "pinned versions. If a package is missing, that's a "
            "structured-error artifact, not a thing to silently upgrade."
        ),
    },
    {
        "kind": "skip_deterministic_leaves",
        "pattern": r"focus\s+on\s+(the\s+)?algorithm|first\s+the\s+(algorithm|implementation|code)|implement\s+first",
        "lesson": (
            "You're saying 'implement the algorithm first' or "
            "'focus on the code'. In paperbench that's wrong: the "
            "deterministic leaves (operational hygiene + artifact "
            "structure) are worth ~70 points and the algorithm is "
            "worth ~25. The 'write metrics.json stub first' / 'read the "
            "rubric first' sequence scores higher than 'tunnel vision "
            "on the algorithm'. Reorder: operational artifacts first, "
            "algorithm second, experiment third."
        ),
    },
    {
        "kind": "ref_author_repo",
        "pattern": r"safe-torch|safe-jax|log-postech|github\.com/LOG-postech",
        "lesson": (
            "You're about to reference the author repository. The "
            "paperbench network blacklist blocks it AND the rubric's "
            "L001 explicitly checks 'no safe-torch or safe-jax import'. "
            "Reconstruct the algorithm from the paper alone — the "
            "oracle provenance shows it's possible to do this in "
            "~240 minutes of expert time without the repo."
        ),
    },
    {
        "kind": "arxiv_reference",
        "pattern": r"arxiv\.org/(abs|pdf|e-print)/25",
        "lesson": (
            "You're about to reference an arxiv URL. The paperbench "
            "network blacklist blocks it AND the rubric's L001 "
            "checks 'does not access arxiv / openreview / paperswithcode "
            "URLs in the source'. Use the paper artifact in "
            "/workspace/paper/ — it has the full text + extracted "
            "sections."
        ),
    },
]


def get_catches() -> list[dict[str, str]]:
    """Return the active catch list, paperbench-augmented if applicable.

    Paperbench mode is enabled by setting ``MIANMI_HEADLESS_PAPERBENCH=1``.
    We do NOT auto-detect from ``/workspace/submission/tests/rubrics.json``
    because that file is a verifier artifact, not an agent artifact.
    """
    catches: list[dict[str, str]] = list(SEED_CATCHES)
    if os.environ.get("MIANMI_HEADLESS_PAPERBENCH") == "1":
        catches = list(SEED_CATCHES) + list(PAPERBENCH_CATCHES)
    return catches


# --------------------------------------------------------------------------- #
# Toll result + singleton
# --------------------------------------------------------------------------- #

@dataclass
class TollFired:
    """The troll's response when a pattern matches."""
    kind: str
    pattern: str
    lesson: str
    matched_text: str
    tool_name: str
    tool_input: Any

    def to_prompt(self) -> str:
        return (
            f"## Troll toll fired\n\n"
            f"**Catch (kind={self.kind!r}):** matched `{self.pattern}` "
            f"in your reasoning: _{self.matched_text[:120]}_.\n\n"
            f"**Lesson:** {self.lesson}\n\n"
            f"**Before you call `{self.tool_name}` with this input, you must do one of:**\n\n"
            f"1. **Run a web search** to verify the information is current "
            f"(use the `scrape_openrouter_models` tool or any web search).\n"
            f"2. **Rewrite the call** to avoid the pattern.\n"
            f"3. **Defend the call** in ONE sentence — if you can justify it, "
            f"the toll is waived for this call.\n"
            f"4. **Escalate to the user** if you're genuinely stuck.\n\n"
            f"Pick one. The toll is not a hard block — it's an annoying "
            f"reminder to slow down before you ship something stupid."
        )


def _pattern_matches(pattern: str, haystack: str) -> bool:
    if not pattern:
        return False
    p = pattern.lower()
    h = haystack.lower()
    has_metachars = any(c in pattern for c in r".*+?^$()[]{}|\\")
    if has_metachars:
        try:
            return bool(re.search(p, h))
        except re.error:
            return p in h
    return p in h


class TrollToll:
    """The headless troll. No DB, no async, no metrics. Just the seed list."""

    def __init__(self, seed_catches: list[dict[str, str]] | None = None):
        # If no explicit list is given, use the auto-detected list
        # (which is paperbench-augmented if we're inside a paperbench trial).
        if seed_catches is None:
            self.catches = get_catches()
        else:
            self.catches = list(seed_catches)

    def check(
        self,
        *,
        reasoning: str = "",
        tool_name: str = "",
        tool_input: Any = None,
    ) -> TollFired | None:
        """Return a TollFired if any catch matches, else None.

        The haystack is reasoning + tool_name + stringified(tool_input),
        case-insensitive. Regex metachars in the pattern trigger regex
        matching; otherwise plain substring.
        """
        import json
        if not self.catches:
            return None
        parts = [reasoning, tool_name]
        if tool_input is not None:
            if isinstance(tool_input, str):
                parts.append(tool_input)
            else:
                try:
                    parts.append(json.dumps(tool_input, default=str))
                except Exception:
                    parts.append(str(tool_input))
        haystack = "\n".join(p for p in parts if p)
        if not haystack:
            return None

        for catch in self.catches:
            if _pattern_matches(catch["pattern"], haystack):
                # Find the matched text for the lesson prompt
                idx = haystack.lower().find(catch["pattern"].lower())
                matched = haystack[idx:idx + len(catch["pattern"])] if idx >= 0 else catch["pattern"]
                return TollFired(
                    kind=catch["kind"],
                    pattern=catch["pattern"],
                    lesson=catch["lesson"],
                    matched_text=matched,
                    tool_name=tool_name,
                    tool_input=tool_input,
                )
        return None
