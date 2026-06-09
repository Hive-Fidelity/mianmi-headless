"""Tests for the ATIF trajectory conversion.

These tests build a fake event log (mimicking what the headless agent
emits during a real run), feed it to ``events_to_trajectory``, and
verify the resulting ATIF dict matches the harbor spec.

Run with:  pytest tests/test_atif.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_events():
    """A typical run: user message, reasoning, assistant text, tool call,
    tool result, then a second iteration with reasoning + text.

    Mirrors what the headless agent actually emits.
    """
    return [
        {
            "kind": "user",
            "turn_number": 2,
            "timestamp": "2026-06-08T18:35:09Z",
            "text": "What files are in this directory?",
        },
        {
            "kind": "reasoning",
            "turn_number": 3,
            "timestamp": "2026-06-08T18:35:11Z",
            "text": "Let me check using list_files.",
        },
        {
            "kind": "assistant_text",
            "turn_number": 3,
            "timestamp": "2026-06-08T18:35:11Z",
            "text": "",
            "metrics": {
                "prompt_tokens": 100,
                "completion_tokens": 22,
                "cached_tokens": 50,
                "cost_usd": 0.0003,
            },
        },
        {
            "kind": "tool_call",
            "turn_number": 3,
            "timestamp": "2026-06-08T18:35:11Z",
            "tool_call_id": "call-1",
            "tool_name": "list_files",
            "arguments": {"pattern": "*"},
        },
        {
            "kind": "tool_result",
            "turn_number": 4,
            "timestamp": "2026-06-08T18:35:12Z",
            "source_call_id": "call-1",
            "tool_name": "list_files",
            "content": "Files:\n  turns.jsonl  1.0 KB",
        },
        {
            "kind": "reasoning",
            "turn_number": 5,
            "timestamp": "2026-06-08T18:35:13Z",
            "text": "Now I know the answer.",
        },
        {
            "kind": "assistant_text",
            "turn_number": 5,
            "timestamp": "2026-06-08T18:35:13Z",
            "text": "There is 1 file: turns.jsonl.",
            "metrics": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "cached_tokens": 50,
                "cost_usd": 0.0004,
            },
        },
    ]


# --------------------------------------------------------------------------- #
# Basic conversion
# --------------------------------------------------------------------------- #

class TestAtifBasic:
    def test_schema_version_is_v15(self, sample_events):
        from mianmi_headless.atif import events_to_trajectory
        traj = events_to_trajectory(sample_events)
        assert traj["schema_version"] == "ATIF-v1.5"

    def test_session_id_set(self, sample_events):
        from mianmi_headless.atif import events_to_trajectory
        traj = events_to_trajectory(sample_events)
        assert "session_id" in traj
        assert traj["session_id"] is not None
        assert "trajectory_id" in traj

    def test_step_ids_are_sequential(self, sample_events):
        from mianmi_headless.atif import events_to_trajectory
        traj = events_to_trajectory(sample_events)
        step_ids = [s["step_id"] for s in traj["steps"]]
        assert step_ids == list(range(1, len(traj["steps"]) + 1))

    def test_agent_block(self, sample_events):
        from mianmi_headless.atif import events_to_trajectory
        traj = events_to_trajectory(
            sample_events, agent_name="mianmi-headless", model_name="gpt-5.5",
        )
        assert traj["agent"]["name"] == "mianmi-headless"
        assert traj["agent"]["version"] == "0.1.0"
        assert traj["agent"]["model_name"] == "gpt-5.5"

    def test_user_step(self, sample_events):
        from mianmi_headless.atif import events_to_trajectory
        traj = events_to_trajectory(sample_events)
        user_step = traj["steps"][0]
        assert user_step["source"] == "user"
        assert user_step["message"] == "What files are in this directory?"

    def test_agent_step_has_model_name(self, sample_events):
        from mianmi_headless.atif import events_to_trajectory
        traj = events_to_trajectory(sample_events, model_name="gpt-5.5")
        # The pure-LLM agent steps (no observation) carry the model
        # name. The observation step is also source=agent but it
        # represents a tool result, not an LLM call.
        llm_steps = [
            s for s in traj["steps"]
            if s["source"] == "agent" and s.get("observation") is None
        ]
        assert len(llm_steps) >= 1
        for s in llm_steps:
            assert s["model_name"] == "gpt-5.5"

    def test_tool_call_step(self, sample_events):
        from mianmi_headless.atif import events_to_trajectory
        traj = events_to_trajectory(sample_events)
        # Find the agent step that has tool_calls
        tool_step = next(
            s for s in traj["steps"]
            if s.get("source") == "agent" and s.get("tool_calls")
        )
        assert len(tool_step["tool_calls"]) == 1
        tc = tool_step["tool_calls"][0]
        assert tc["tool_call_id"] == "call-1"
        assert tc["function_name"] == "list_files"
        assert tc["arguments"] == {"pattern": "*"}

    def test_observation_step_has_source_call_id(self, sample_events):
        from mianmi_headless.atif import events_to_trajectory
        traj = events_to_trajectory(sample_events)
        obs_step = next(
            s for s in traj["steps"]
            if s.get("source") == "agent" and s.get("observation")
        )
        results = obs_step["observation"]["results"]
        assert len(results) == 1
        assert results[0]["source_call_id"] == "call-1"
        assert "turns.jsonl" in results[0]["content"]


# --------------------------------------------------------------------------- #
# Final metrics
# --------------------------------------------------------------------------- #

class TestFinalMetrics:
    def test_final_metrics_aggregate(self, sample_events):
        from mianmi_headless.atif import events_to_trajectory
        traj = events_to_trajectory(sample_events)
        fm = traj["final_metrics"]
        # Two assistant_text events with metrics: 100+120 = 220 prompt
        assert fm["total_prompt_tokens"] == 220
        # 22+30 = 52 completion
        assert fm["total_completion_tokens"] == 52
        # 50+50 = 100 cached
        assert fm["total_cached_tokens"] == 100
        # 0.0003+0.0004 = 0.0007
        assert abs(fm["total_cost_usd"] - 0.0007) < 0.0001
        # total_steps counts ALL steps (including user and observation)
        assert fm["total_steps"] == len(traj["steps"])


# --------------------------------------------------------------------------- #
# Sub-agent trajectories (gardener)
# --------------------------------------------------------------------------- #

class TestSubagentTrajectories:
    def test_gardener_call_emits_subagent_trajectory(self):
        from mianmi_headless.atif import events_to_trajectory
        events = [
            {
                "kind": "user", "turn_number": 2,
                "timestamp": "2026-06-08T18:35:09Z",
                "text": "What did we do in turn 5?",
            },
            {
                "kind": "reasoning", "turn_number": 3,
                "timestamp": "2026-06-08T18:35:10Z",
                "text": "I'll ask the gardener.",
            },
            {
                "kind": "assistant_text", "turn_number": 3,
                "timestamp": "2026-06-08T18:35:10Z",
                "text": "",
            },
            {
                "kind": "tool_call", "turn_number": 3,
                "timestamp": "2026-06-08T18:35:10Z",
                "tool_call_id": "call-g1",
                "tool_name": "ask_gardener",
                "arguments": {"query": "What did we do in turn 5?"},
            },
            {
                "kind": "tool_result", "turn_number": 4,
                "timestamp": "2026-06-08T18:35:11Z",
                "source_call_id": "call-g1",
                "tool_name": "ask_gardener",
                "content": "In turn 5 you wrote tests.",
                "is_subagent": True,
                "subagent_trajectory": {
                    "schema_version": "ATIF-v1.5",
                    "session_id": "test-session",
                    "agent": {
                        "name": "mianmi-gardener",
                        "version": "0.1.0",
                        "model_name": "MiniMax-M3",
                    },
                    "steps": [
                        {"step_id": 1, "source": "user",
                         "message": "What did we do in turn 5?"},
                        {"step_id": 2, "source": "agent",
                         "message": "In turn 5 you wrote tests.",
                         "model_name": "MiniMax-M3", "llm_call_count": 1},
                    ],
                },
            },
        ]
        traj = events_to_trajectory(events, session_id="test-session")
        # Sub-agent trajectory is embedded
        assert "subagent_trajectories" in traj
        assert len(traj["subagent_trajectories"]) == 1
        sub = traj["subagent_trajectories"][0]
        assert sub["agent"]["name"] == "mianmi-gardener"
        assert sub["agent"]["model_name"] == "MiniMax-M3"
        # The observation in the parent has a SubagentTrajectoryRef
        obs_step = next(
            s for s in traj["steps"]
            if s.get("source") == "agent" and s.get("observation")
        )
        result = obs_step["observation"]["results"][0]
        assert "subagent_trajectory_ref" in result
        ref = result["subagent_trajectory_ref"][0]
        assert ref["trajectory_id"] is not None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

class TestValidation:
    def test_valid_trajectory_passes_validation(self, sample_events):
        from mianmi_headless.atif import events_to_trajectory, validate_against_atif
        traj = events_to_trajectory(sample_events)
        errors = validate_against_atif(traj)
        assert errors == [], f"unexpected errors: {errors}"

    def test_step_ids_out_of_order_fails_validation(self):
        from mianmi_headless.atif import events_to_trajectory, validate_against_atif
        traj = events_to_trajectory([])
        # Manually craft a bad trajectory
        bad = {
            "schema_version": "ATIF-v1.5",
            "agent": {"name": "x", "version": "1"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "hi"},
                {"step_id": 5, "source": "user", "message": "again"},  # wrong
            ],
        }
        errors = validate_against_atif(bad)
        assert any("step_id" in e for e in errors)

    def test_observation_referencing_missing_call_fails(self):
        from mianmi_headless.atif import validate_against_atif
        bad = {
            "schema_version": "ATIF-v1.5",
            "agent": {"name": "x", "version": "1"},
            "steps": [
                {"step_id": 1, "source": "agent", "message": "",
                 "tool_calls": [{"tool_call_id": "a", "function_name": "x", "arguments": {}}],
                 "observation": {"results": [
                     {"source_call_id": "b", "content": "x"}  # 'b' not in tool_calls
                 ]}},
            ],
        }
        errors = validate_against_atif(bad)
        assert any("source_call_id" in e for e in errors)

    def test_agent_only_field_on_user_step_fails(self):
        from mianmi_headless.atif import validate_against_atif
        bad = {
            "schema_version": "ATIF-v1.5",
            "agent": {"name": "x", "version": "1"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "hi",
                 "model_name": "gpt-5.5"},  # agent-only field on user step
            ],
        }
        errors = validate_against_atif(bad)
        assert any("model_name" in e for e in errors)


# --------------------------------------------------------------------------- #
# End-to-end: EventLog → emit_trajectory
# --------------------------------------------------------------------------- #

class TestEndToEnd:
    def test_eventlog_to_trajectory(self, tmp_path):
        """Write events to an EventLog, then convert to ATIF."""
        from mianmi_headless.events import (
            EventLog, UserEvent, ReasoningEvent, AssistantTextEvent,
            ToolCallEvent, ToolResultEvent,
        )
        from mianmi_headless.atif import events_to_trajectory, validate_against_atif
        log_path = tmp_path / "events.jsonl"
        elog = EventLog(log_path)
        elog.append(UserEvent(turn_number=1, timestamp="2026-06-08T19:00:00Z", text="hi"))
        elog.append(ReasoningEvent(turn_number=2, timestamp="2026-06-08T19:00:01Z", text="thinking"))
        elog.append(AssistantTextEvent(
            turn_number=2, timestamp="2026-06-08T19:00:01Z",
            text="hello", metrics={"prompt_tokens": 10, "completion_tokens": 5},
        ))
        elog.close()

        events = list(elog.iter_events())
        traj = events_to_trajectory(events, model_name="gpt-5.5", session_id="sess-1")
        errors = validate_against_atif(traj)
        assert errors == [], errors
        # 1 user step + 1 agent step (reasoning+text merged)
        assert len(traj["steps"]) == 2
        # Session id preserved
        assert traj["session_id"] == "sess-1"
        # Trajectory id set
        assert traj["trajectory_id"] == "sess-1"
