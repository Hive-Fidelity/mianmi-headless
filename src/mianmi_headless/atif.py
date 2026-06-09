"""Convert the headless agent's event log into an ATIF-v1.5 trajectory.

ATIF (Agent Trajectory Interchange Format) is harbor's trajectory
spec. The full spec lives in `harbor.models.trajectories`. We don't
import the harbor models (heavy dep) — we emit plain dicts that
match the ATIF-v1.5 schema exactly. Harbor's validator can re-parse
them.

Why ATIF-v1.5 and not v1.7? v1.7 added the ``subagent_trajectories``
array, ``llm_call_count``, and ``trajectory_id`` — we support those
in spirit (gardener calls emit subagent refs, llm_call_count defaults
to 1, trajectory_id is set to the session id) but emit ``ATIF-v1.5``
in the ``schema_version`` field for maximum compatibility with the
existing harbor validator. The new fields go in ``extra`` if needed.

The conversion is a single function: ``events_to_trajectory(events,
agent_info) -> dict``. The dict is what ``populate_context_post_run``
writes to ``trajectory.json`` in the agent dir.

Subagent support: when a tool result was produced by a sub-agent
(``is_subagent=True``), the converter emits the sub-agent's
trajectory as an embedded ``subagent_trajectories`` entry and adds a
``SubagentTrajectoryRef`` to the parent's ``ObservationResult``. This
is exactly how harbor's Codex agent handles web_search_call
delegation. The gardener's trajectory lives in here.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

log = logging.getLogger("mianmi_headless.atif")


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def events_to_trajectory(
    events: list[dict],
    *,
    agent_name: str = "mianmi-headless",
    agent_version: str = "0.1.0",
    model_name: str | None = None,
    tool_definitions: list[dict] | None = None,
    reasoning_effort: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Build an ATIF-v1.5 Trajectory dict from a list of events.

    Events are produced by ``EventLog.iter_events()`` and come from
    the headless agent's run.

    Returns a dict ready to be JSON-serialized as ``trajectory.json``.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Aggregate metrics across all assistant turns.
    total_prompt = 0
    total_completion = 0
    total_cached = 0
    total_cost = 0.0

    # First pass: collect cost from events. We need a richer event
    # payload (with usage) for accurate cost tracking. The agent
    # loop emits that — see ``_emit_iteration_metrics`` in
    # ``agent.py``.
    for ev in events:
        if ev.get("kind") == "assistant_text" or ev.get("kind") == "reasoning":
            m = ev.get("metrics") or {}
            total_prompt += m.get("prompt_tokens", 0) or 0
            total_completion += m.get("completion_tokens", 0) or 0
            total_cached += m.get("cached_tokens", 0) or 0
            total_cost += m.get("cost_usd", 0) or 0

    # Build steps. Each step is one of:
    #   - user message
    #   - agent turn: text + (optional) reasoning + (optional) tool calls
    #   - observation: tool results
    # We group consecutive agent events (reasoning, text, tool_call)
    # into a single Step so the trajectory reads naturally.
    steps: list[dict] = []
    subagent_trajectories: list[dict] = []

    i = 0
    step_id = 0
    while i < len(events):
        ev = events[i]
        kind = ev.get("kind")

        if kind == "user":
            step_id += 1
            steps.append({
                "step_id": step_id,
                "timestamp": ev.get("timestamp"),
                "source": "user",
                "message": ev.get("text", ""),
            })
            i += 1

        elif kind in ("reasoning", "assistant_text", "tool_call"):
            # Collect all consecutive agent events into one Step.
            step_id += 1
            message_text = ""
            reasoning_text = ""
            tool_calls: list[dict] = []
            metrics: dict | None = None
            step_ts = ev.get("timestamp")

            while i < len(events) and events[i].get("kind") in (
                "reasoning", "assistant_text", "tool_call",
            ):
                ev_i = events[i]
                if ev_i.get("kind") == "reasoning":
                    reasoning_text = ev_i.get("text", "")
                elif ev_i.get("kind") == "assistant_text":
                    message_text = ev_i.get("text", "")
                    metrics = ev_i.get("metrics")
                elif ev_i.get("kind") == "tool_call":
                    tool_calls.append({
                        "tool_call_id": ev_i.get("tool_call_id", ""),
                        "function_name": ev_i.get("tool_name", ""),
                        "arguments": ev_i.get("arguments", {}) or {},
                    })
                i += 1

            step: dict = {
                "step_id": step_id,
                "timestamp": step_ts,
                "source": "agent",
                "message": message_text or "",
            }
            if model_name:
                step["model_name"] = model_name
            if reasoning_effort:
                step["reasoning_effort"] = reasoning_effort
            if reasoning_text:
                step["reasoning_content"] = reasoning_text
            if tool_calls:
                step["tool_calls"] = tool_calls
            if metrics:
                # ATIF Metrics schema
                step["metrics"] = _build_metrics(metrics)
            step["llm_call_count"] = 1  # each step = 1 LLM inference
            steps.append(step)

        elif kind == "tool_result":
            # Find any preceding tool_call in the same Step and attach
            # the result as an Observation. If the tool was a sub-agent
            # (the gardener), embed the sub-agent's trajectory.
            step_id += 1
            source_call_id = ev.get("source_call_id", "")
            content = ev.get("content", "")
            is_subagent = ev.get("is_subagent", False)
            sub_traj = ev.get("subagent_trajectory")
            obs_results: list[dict] = [{
                "source_call_id": source_call_id or None,
                "content": content,
            }]
            if is_subagent and sub_traj:
                sub_traj_id = sub_traj.get("trajectory_id") or str(uuid.uuid4())
                obs_results[0]["subagent_trajectory_ref"] = [{
                    "trajectory_id": sub_traj_id,
                    "trajectory_path": None,  # embedded
                }]
                # Make sure the embedded trajectory has its own id.
                sub_traj["trajectory_id"] = sub_traj_id
                if "session_id" not in sub_traj:
                    sub_traj["session_id"] = session_id
                subagent_trajectories.append(sub_traj)
            steps.append({
                "step_id": step_id,
                "timestamp": ev.get("timestamp"),
                "source": "agent",
                "message": "",
                "observation": {"results": obs_results},
                "llm_call_count": 0,  # no LLM call — just a tool result
            })
            i += 1

        else:
            # Unknown event kind — skip rather than crash
            log.warning("unknown event kind %r, skipping", kind)
            i += 1

    final_metrics: dict = {
        "total_prompt_tokens": total_prompt or None,
        "total_completion_tokens": total_completion or None,
        "total_cached_tokens": total_cached or None,
        "total_cost_usd": total_cost or None,
        "total_steps": len(steps),
    }

    trajectory: dict = {
        "schema_version": "ATIF-v1.5",
        "session_id": session_id,
        "trajectory_id": session_id,
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": model_name,
        },
        "steps": steps,
        "final_metrics": final_metrics,
    }
    if tool_definitions:
        trajectory["agent"]["tool_definitions"] = tool_definitions
    if subagent_trajectories:
        trajectory["subagent_trajectories"] = subagent_trajectories
    return trajectory


def _build_metrics(m: dict) -> dict:
    """Convert our internal metrics dict to ATIF Metrics schema."""
    out: dict = {}
    if (v := m.get("prompt_tokens")) is not None:
        out["prompt_tokens"] = v
    if (v := m.get("completion_tokens")) is not None:
        out["completion_tokens"] = v
    if (v := m.get("cached_tokens")) is not None:
        out["cached_tokens"] = v
    if (v := m.get("cost_usd")) is not None:
        out["cost_usd"] = v
    return out


# --------------------------------------------------------------------------- #
# Self-validation (best-effort)
# --------------------------------------------------------------------------- #

def validate_against_atif(trajectory: dict) -> list[str]:
    """Check the trajectory matches ATIF-v1.5 conventions. Returns
    a list of error messages (empty = OK). Doesn't enforce every
    rule — just the ones the headless harness is likely to break.
    """
    errors: list[str] = []
    if trajectory.get("schema_version") not in (
        "ATIF-v1.0", "ATIF-v1.1", "ATIF-v1.2", "ATIF-v1.3",
        "ATIF-v1.4", "ATIF-v1.5", "ATIF-v1.6", "ATIF-v1.7",
    ):
        errors.append(f"invalid schema_version: {trajectory.get('schema_version')!r}")
    steps = trajectory.get("steps", [])
    if not steps:
        errors.append("trajectory has no steps")
    for i, step in enumerate(steps):
        expected_id = i + 1
        if step.get("step_id") != expected_id:
            errors.append(
                f"step {i} has step_id={step.get('step_id')!r}, expected {expected_id}"
            )
        if step.get("source") not in ("system", "user", "agent"):
            errors.append(f"step {i} has invalid source: {step.get('source')!r}")
        if "message" not in step:
            errors.append(f"step {i} missing 'message'")
        # Agent-only fields
        if step.get("source") != "agent":
            for f in ("model_name", "reasoning_effort", "reasoning_content",
                      "tool_calls", "metrics"):
                if step.get(f) is not None:
                    errors.append(
                        f"step {i} has agent-only field {f!r} but source={step.get('source')!r}"
                    )
        # Tool call ↔ observation pairing
        if step.get("tool_calls"):
            tc_ids = {tc.get("tool_call_id") for tc in step["tool_calls"]}
            obs = step.get("observation", {})
            for r in obs.get("results", []):
                src = r.get("source_call_id")
                if src is not None and src not in tc_ids:
                    errors.append(
                        f"step {i} observation references source_call_id={src!r} "
                        f"not in tool_calls ({sorted(tc_ids)!r})"
                    )
    return errors
