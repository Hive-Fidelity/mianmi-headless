"""Raw turn log — append-only JSONL on disk.

The single source of truth for the agent's conversation. Every turn
the agent sees (user, assistant, tool result) is appended here in
temporal order, never compacted.

This module is dependency-light on purpose — no agent code, no tools,
no API clients — so it can be imported from anywhere without
circular-import risk.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

log = logging.getLogger("mianmi_headless.turn_log")


class Turn(BaseModel):
    """A single turn in the raw turn log.

    Mirrors the full mianmi project's turns table, simplified to
    JSONL on disk. The agent writes one of these per response cycle.
    """

    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turn_number: int
    role: str  # "user" | "assistant" | "tool" | "system"
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    content: Any  # string for user/assistant text, dict for tool results
    # Optional bookkeeping
    tool_name: str | None = None
    tool_call_id: str | None = None
    reasoning: str | None = None
    # Accept either a flat dict (e.g. {"input_tokens": 5148, ...}) or a
    # nested usage object (e.g. ResponseUsage with input_tokens_details
    # etc.). We normalize on the way in.
    usage: dict[str, Any] | None = None

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> "Turn":
        return cls.model_validate_json(line)


class TurnLog:
    """Append-only JSONL file. One turn per line, in temporal order.

    Designed for hours-long runs — every turn the agent sees is
    persisted before the next iteration. The file is the source of
    truth; the agent's in-memory state is just a recent slice.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Don't truncate on open — we're append-only.
        self._fp = open(self.path, "a", encoding="utf-8", buffering=1)
        self._counter = self._recover_count()

    def _recover_count(self) -> int:
        """If the file exists with prior turns, find the max turn_number
        so we don't restart from 1 on resume.
        """
        if not self.path.exists():
            return 0
        max_n = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = Turn.from_jsonl(line)
                    max_n = max(max_n, t.turn_number)
                except Exception:
                    log.warning("skipped malformed turn line: %r", line[:80])
        return max_n

    @property
    def turn_count(self) -> int:
        return self._counter

    def append(self, turn: Turn) -> None:
        self._fp.write(turn.to_jsonl() + "\n")
        self._fp.flush()
        self._counter = max(self._counter, turn.turn_number)

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def iter_turns(self, start: int | None = None, end: int | None = None) -> Iterator[Turn]:
        """Iterate turns in temporal order, optionally bounded by turn_number."""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = Turn.from_jsonl(line)
                except Exception:
                    continue
                if start is not None and t.turn_number < start:
                    continue
                if end is not None and t.turn_number > end:
                    continue
                yield t

    def __enter__(self) -> "TurnLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
