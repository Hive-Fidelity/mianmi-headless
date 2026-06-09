"""Structured event log for the headless agent.

The raw turn log (`turn_log.py`) is the immutable source of truth for
the agent's conversation. This module adds a parallel **structured
event log** that captures the same information in a form that's
trivially convertible to harbor's ATIF trajectory format.

Why both? The raw turn log is for the gardener + human debugging —
it's the source of truth, in human-readable JSONL. The event log is
for the ATIF converter — it's also JSONL, but the entries are typed
events (``user_message``, ``reasoning``, ``tool_call``, ``tool_result``,
``assistant_text``) that map 1:1 to ATIF ``Step`` fields.

Keeping them separate means we can change the ATIF format (or add
subagent trajectories, etc.) without disturbing the raw turn stream.

Event log format (one JSON object per line):

    {
      "kind": "user" | "assistant_text" | "reasoning" | "tool_call" | "tool_result",
      "turn_number": int,
      "timestamp": "2026-06-08T...",
      ...kind-specific fields...
    }
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("mianmi_headless.events")


# --------------------------------------------------------------------------- #
# Event types
# --------------------------------------------------------------------------- #

@dataclass
class UserEvent:
    kind: str = "user"
    turn_number: int = 0
    timestamp: str = ""
    text: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class AssistantTextEvent:
    kind: str = "assistant_text"
    turn_number: int = 0
    timestamp: str = ""
    text: str = ""
    # Iteration metrics (prompt_tokens, completion_tokens, etc.) so
    # the ATIF converter can build the per-step Metrics block.
    metrics: dict | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class ReasoningEvent:
    kind: str = "reasoning"
    turn_number: int = 0
    timestamp: str = ""
    text: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class ToolCallEvent:
    kind: str = "tool_call"
    turn_number: int = 0
    timestamp: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class ToolResultEvent:
    """A tool execution result. May be from a local tool (read_file, etc.)
    or from a sub-agent (the gardener). ``source_call_id`` ties this
    back to a ``ToolCallEvent``.
    """
    kind: str = "tool_result"
    turn_number: int = 0
    timestamp: str = ""
    source_call_id: str = ""
    tool_name: str = ""
    content: str = ""
    # If this result is from a sub-agent (e.g. the gardener), we
    # embed the sub-agent's trajectory here. The ATIF converter
    # will emit a SubagentTrajectoryRef + an embedded
    # subagent_trajectories entry.
    is_subagent: bool = False
    subagent_trajectory: dict | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


# Union type
Event = UserEvent | AssistantTextEvent | ReasoningEvent | ToolCallEvent | ToolResultEvent


# --------------------------------------------------------------------------- #
# Event log
# --------------------------------------------------------------------------- #

class EventLog:
    """Append-only JSONL of structured events. Mirrors the turn log
    but with typed events suitable for ATIF conversion.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8", buffering=1)
        self._counter = 0

    def append(self, event: Event) -> None:
        self._fp.write(event.to_json() + "\n")
        self._fp.flush()
        self._counter += 1

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def iter_events(self) -> Iterator[dict]:
        """Iterate all events as raw dicts. Skips malformed lines.
        Used by the ATIF converter.
        """
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipped malformed event line: %r", line[:80])

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
