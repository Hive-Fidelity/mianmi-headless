"""mianmi-headless — the headless mianmi agent for harbor/terminal-bench.

Design goals (in order of importance):
  1. Predictable. The agent does what it says it will do. No silent
     truncations, no provider fallback, no "we cut you off to save
     you money" lies.
  2. Long-context. 1M tokens, hard. truncation='disabled' on the
     Responses API. Loud failure on overflow.
  3. Stateful. Every turn is persisted to a raw turn log
     (``turns.jsonl``). The gardener reads from this log when the
     main agent needs historical context.
  4. Self-aware. The troll toll catches known-bad agent behavior
     (stale model ids, "the latest model" phrasing) before the
     call goes out.
  5. Single-process. No server, no DB, no message bus. Just a
     Python CLI that harbor can ``pip install`` and run.
"""

from __future__ import annotations

__version__ = "0.3.0"
