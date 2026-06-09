"""The agent core.

This is the heart of the headless harness. A ``HeadlessAgent`` instance
holds:

* A raw turn log (JSONL on disk) — every turn the agent sees, written
  in temporal order, never compacted.
* A client to the OpenAI Responses API.
* A client to the M3 gardener.
* A set of tools the agent can call.
* A pre-tool-call troll toll.

The agent runs an **agentic loop** that:
  1. Sends the current turn + raw turn log to the Responses API.
  2. If the response has tool calls, dispatches them, appends the
     results to the log, and loops.
  3. When the response has no tool calls, returns the final text.

The loop is intentionally simple — no streaming, no async choreography.
The user's benchmark tasks run for hours; we want the loop to be
boring, debuggable, and total-order correct.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field

from mianmi_headless.atif import events_to_trajectory, validate_against_atif
from mianmi_headless.events import (
    AssistantTextEvent,
    EventLog,
    ReasoningEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserEvent,
)
from mianmi_headless.paperbench import paperbench_system_prompt
from mianmi_headless.tools import ToolContext, registry as default_tool_registry
from mianmi_headless.troll import SEED_CATCHES, TrollToll
from mianmi_headless.turn_log import Turn, TurnLog

log = logging.getLogger("mianmi_headless.agent")


def _default_main_model() -> str:
    """Resolve the main model name.

    Priority: explicit ``MIANMI_HEADLESS_MODEL`` env var > paperbench
    mode (gpt-5.5-pro) > base default (gpt-5.5).

    Paperbench tasks are long-running, multi-hour, research-grade —
    worth the extra cost of gpt-5.5-pro. Non-paperbench tasks
    (interactive shell, dev) stay on gpt-5.5 to keep costs down.

    Note: we do NOT auto-detect paperbench mode by looking for
    ``/workspace/submission/tests/rubrics.json`` — that file is a
    verifier artifact, not an agent artifact, and seeing it would
    be a form of cheating. The user must explicitly set
    ``MIANMI_HEADLESS_PAPERBENCH=1`` (or the harbor patch can do it).
    """
    explicit = os.getenv("MIANMI_HEADLESS_MODEL")
    if explicit:
        return explicit
    if os.getenv("MIANMI_HEADLESS_PAPERBENCH") == "1":
        return "gpt-5.5-pro"
    return "gpt-5.5"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

class AgentConfig(BaseModel):
    """All knobs the headless agent exposes.

    Defaults are designed for the harbor benchmark case:
      * 1M context (gpt-5.5 max)
      * high reasoning effort
      * truncation disabled (we want loud failure, not silent)
      * no fallback (the user wants to know if the API is down)
    """

    # --- Main model (the agent's brain) -------------------------------- #
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None
    )
    # Model default. The base default is gpt-5.5; for paperbench
    # mode (auto-detected) we default to gpt-5.5-pro for more
    # capability on long-running research tasks.
    main_model: str = Field(
        default_factory=lambda: _default_main_model()
    )
    reasoning_effort: str = Field(default_factory=lambda: os.getenv("MIANMI_HEADLESS_REASONING", "high"))
    # truncation='disabled' is the headline feature of this harness.
    # If the context overflows, the API returns an error. We surface
    # it to the user instead of pretending the agent is still working.
    truncation: str = "disabled"

    # --- Gardener model (the M3 sidecar) ------------------------------- #
    gardener_enabled: bool = Field(
        default_factory=lambda: os.getenv("MIANMI_HEADLESS_GARDENER", "1") != "0"
    )
    minimax_api_key: str = Field(
        default_factory=lambda: os.getenv("MINIMAX_SUBSCRIBER_KEY", "")
    )
    minimax_base_url: str = Field(
        default_factory=lambda: os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
    )
    minimax_tier: str = Field(default_factory=lambda: os.getenv("MINIMAX_TIER", "priority"))
    gardener_model: str = "MiniMax-M3"

    # --- Turn log ------------------------------------------------------ #
    # Where to write raw turns. harbor sets this to the task's working dir.
    turn_log_path: Path = Field(
        default_factory=lambda: Path(os.getenv("MIANMI_HEADLESS_TURN_LOG", "./turns.jsonl"))
    )

    # --- Agent loop budget -------------------------------------------- #
    # Cap on tool-call iterations per turn. Defends against infinite
    # tool-call loops when something goes wrong.
    max_tool_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MIANMI_HEADLESS_MAX_ITER", "200"))
    )
    # Per-iteration cap on tool output size. Keeps the agent from
    # accidentally slurping 50MB of stdout into its context.
    max_tool_output_chars: int = 200_000

    # --- Troll toll --------------------------------------------------- #
    troll_enabled: bool = Field(
        default_factory=lambda: os.getenv("MIANMI_HEADLESS_TROLL", "1") != "0"
    )

    # --- Codex-native tool types -------------------------------------- #
    # We pass these as `tools=` to the Responses API. Default is the
    # file_search + code_execution bundle; users can override.
    enable_web_search: bool = True
    enable_code_execution: bool = False  # off by default — harbor envs vary
    enable_file_search: bool = False

    def model_post_init(self, __context) -> None:
        # Validate at construction — fail loud, not on first API call.
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. The headless harness refuses to "
                "run without an explicit key — no 'oops, you forgot' surprises."
            )
        if self.gardener_enabled and not self.minimax_api_key:
            log.warning(
                "Gardener enabled but MINIMAX_SUBSCRIBER_KEY is not set. "
                "The ask_gardener tool will return an error when called. "
                "Either set the key or set MIANMI_HEADLESS_GARDENER=0."
            )


# --------------------------------------------------------------------------- #
# OpenAI Responses API client
# --------------------------------------------------------------------------- #

def _to_response_input(
    turns: list[Turn],
    user_instruction: str | None = None,
) -> list[dict]:
    """Build the Responses API ``input`` list from a slice of turns.

    The Responses API uses ``{"role": ..., "content": ...}`` blocks,
    where ``content`` is a list of typed blocks
    (``{"type": "input_text", "text": ...}``,
    ``{"type": "function_call", ...}``,
    ``{"type": "function_call_output", ...}``, etc.).

    Codex-native tool types are preserved: we pass tool calls back
    to the API with the exact shape the API expects, not whatever
    LiteLLM's chat-completions adapter would re-shape them into.
    """
    out: list[dict] = []
    for t in turns:
        if t.role == "user":
            out.append({
                "role": "user",
                "content": [{"type": "input_text", "text": _text(t.content)}],
            })
        elif t.role == "assistant":
            # Assistant turns may have reasoning + text + tool calls.
            # On the Responses API, function_call and function_call_output
            # are TOP-LEVEL input items, NOT nested in a message's
            # content list. So we split this assistant turn into:
            #   1. An assistant message (with reasoning + text)
            #   2. N top-level function_call items (one per call)
            # Reasoning + text go inside the assistant message.
            text_blocks: list[dict] = []
            if t.reasoning:
                text_blocks.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": t.reasoning}]})
            text = _text(t.content)
            if text:
                text_blocks.append({"type": "output_text", "text": text, "annotations": []})
            if text_blocks:
                out.append({"role": "assistant", "content": text_blocks})
            # Top-level function_call items
            tool_calls_embedded = []
            if isinstance(t.content, dict):
                tool_calls_embedded = t.content.get("tool_calls", []) or []
            for c in tool_calls_embedded:
                out.append({
                    "type": "function_call",
                    "call_id": c.get("call_id", ""),
                    "name": c.get("name", ""),
                    "arguments": c.get("arguments", "{}") or "{}",
                })
        elif t.role == "tool":
            # A tool result. Codex-native type is ``function_call_output``.
            # Top-level item in the input list, paired by call_id with
            # the prior function_call.
            call_id = t.tool_call_id or ""
            output_text = _text(t.content) if not isinstance(t.content, dict) else json.dumps(t.content)
            out.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_text[:200_000],  # hard cap
            })
    if user_instruction is not None:
        out.append({
            "role": "user",
            "content": [{"type": "input_text", "text": user_instruction}],
        })
    return out


def _text(content: Any) -> str:
    """Best-effort stringification of a turn's content field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, default=str)
    return str(content)


# --------------------------------------------------------------------------- #
# Headless agent
# --------------------------------------------------------------------------- #

class HeadlessAgent:
    """The headless mianmi agent.

    Lifecycle:
        config = AgentConfig()
        agent = HeadlessAgent(config)
        result = agent.run("build a transformer from scratch")
        agent.close()
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.turn_log = TurnLog(self.config.turn_log_path)
        # Parallel structured event log for ATIF trajectory conversion.
        # Same lifetime as the turn log; same directory.
        events_path = self.config.turn_log_path.with_name(
            self.config.turn_log_path.stem + ".events.jsonl"
        )
        self.event_log = EventLog(events_path)
        # Session ID is shared between turn log, event log, and
        # ATIF trajectory. Persists across resumes (because the
        # turn log appends; if we ever add resume support, we
        # would read this back).
        self.session_id = str(uuid.uuid4())
        # OpenAI client. Default base URL is the public OpenAI one;
        # users can override (e.g. Azure, an OpenAI-compat proxy) via
        # OPENAI_BASE_URL. We do NOT silently fall back to
        # chat-completions if the Responses API errors — we raise.
        client_kwargs: dict[str, Any] = {"api_key": self.config.openai_api_key}
        if self.config.openai_base_url:
            client_kwargs["base_url"] = self.config.openai_base_url
        self.client = OpenAI(**client_kwargs)
        # Gardener client (httpx — Minimax's OpenAI-compat API).
        self.gardener_http = httpx.Client(
            base_url=self.config.minimax_base_url,
            headers={
                "Authorization": f"Bearer {self.config.minimax_api_key}",
                "X-Tier": self.config.minimax_tier,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
        ) if self.config.gardener_enabled else None
        # Troll toll
        self.troll = (
            TrollToll(seed_catches=SEED_CATCHES) if self.config.troll_enabled else None
        )
        # Tool registry — defaults + the ask_gardener tool
        self.tools = list(default_tool_registry())
        self.tools.append(self._make_ask_gardener_tool())

    # ---- main entry point ------------------------------------------- #

    def run(self, instruction: str) -> str:
        """Run the agent on ``instruction`` and return the final answer.

        The agent runs an iterative tool-call loop until the model
        produces a response with no tool calls. Every iteration
        appends a turn to the raw log.
        """
        log.info("starting headless agent on instruction: %r", instruction[:120])
        log.info("turn log: %s", self.config.turn_log_path)
        log.info("model: %s (1M context, truncation=disabled)", self.config.main_model)

        # Seed the log with a system turn so the gardener + future
        # re-reads know what the task was.
        self.turn_log.append(Turn(
            turn_number=self.turn_log.turn_count + 1,
            role="system",
            content={"text": f"Task: {instruction}\n\nModel: {self.config.main_model}\nGardener: {self.config.gardener_model} (M3, 1M context)"},
        ))

        # The instruction itself becomes the first user turn.
        turn_number = self.turn_log.turn_count + 1
        self.turn_log.append(Turn(
            turn_number=turn_number,
            role="user",
            content={"text": instruction},
        ))
        # Emit a user event for the ATIF trajectory.
        self.event_log.append(UserEvent(
            turn_number=turn_number,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            text=instruction,
        ))

        # Build the Responses API tool list. Codex-native types:
        # function (we register our tools as functions), web_search,
        # file_search, code_interpreter. The API will use the
        # appropriate type for each.
        api_tools = self._build_api_tools()

        # The agentic loop. Each iteration:
        #  1. Send the current slice of turns + the user instruction
        #     to the Responses API.
        #  2. Parse the response. If there are tool calls, dispatch
        #     them, append results to the log, loop.
        #  3. If there are no tool calls, the response is the answer.
        for iteration in range(1, self.config.max_tool_iterations + 1):
            log.debug("agent loop iteration %d", iteration)
            # Build input from the full turn log. The Responses API
            # handles 1M context — that's the whole point of using
            # it raw instead of through LiteLLM.
            input_blocks = _to_response_input(
                list(self.turn_log.iter_turns()),
                user_instruction=None,  # already in the log
            )

            t0 = time.monotonic()
            # Build the Responses API call. The `instructions` field
            # is the system prompt; auto-detect paperbench mode and
            # inject the rubric-aware system prompt if so.
            kwargs: dict = dict(
                model=self.config.main_model,
                input=input_blocks,
                tools=api_tools,
                reasoning={"effort": self.config.reasoning_effort},
                truncation=self.config.truncation,
                store=False,  # we have our own persistent log
            )
            sys_prompt = paperbench_system_prompt()
            if sys_prompt:
                kwargs["instructions"] = sys_prompt
            try:
                response = self.client.responses.create(**kwargs)
            except Exception as e:
                log.exception("Responses API call failed at iteration %d", iteration)
                # Loud failure. Don't try to "fix" it by falling back
                # to a different model or context size. The user
                # explicitly does not want that.
                raise
            latency = time.monotonic() - t0
            log.info(
                "Responses API call: %.1fs, usage=%s",
                latency, getattr(response, "usage", None),
            )

            # Parse the response.
            output_blocks = response.output or []
            tool_calls: list[dict] = []
            text_parts: list[str] = []
            reasoning_text: str | None = None

            for block in output_blocks:
                btype = getattr(block, "type", None)
                if btype == "reasoning":
                    # Codex-native reasoning block. Has summary list.
                    summary = getattr(block, "summary", None) or []
                    parts = [getattr(s, "text", "") for s in summary if getattr(s, "text", None)]
                    if parts:
                        reasoning_text = "\n".join(parts)
                elif btype == "message":
                    # Text content. Codex-native message.
                    for c in getattr(block, "content", []) or []:
                        ctype = getattr(c, "type", None)
                        if ctype in ("output_text", "text"):
                            text_parts.append(getattr(c, "text", ""))
                elif btype == "function_call":
                    # A tool call. Codex-native function_call type.
                    tool_calls.append({
                        "call_id": getattr(block, "call_id", ""),
                        "name": getattr(block, "name", ""),
                        "arguments": getattr(block, "arguments", "{}") or "{}",
                    })
                elif btype in ("web_search_call", "file_search_call", "computer_call"):
                    # Server-side tools — already executed by the API.
                    # We record the call but don't dispatch locally.
                    text_parts.append(f"[{btype}: executed by server]")

            # Persist the assistant turn. The Responses API gives us a
            # Pydantic usage model — flatten it to a dict for the raw
            # turn log (which is JSONL and expects primitives).
            usage_dict: dict[str, Any] | None = None
            if getattr(response, "usage", None) is not None:
                try:
                    raw = response.usage.model_dump()
                    # Flatten: extract the simple counts we care about
                    usage_dict = {
                        "input_tokens": raw.get("input_tokens", 0) or 0,
                        "output_tokens": raw.get("output_tokens", 0) or 0,
                        "total_tokens": raw.get("total_tokens", 0) or 0,
                    }
                    if isinstance(raw.get("input_tokens_details"), dict):
                        usage_dict["cached_tokens"] = (
                            raw["input_tokens_details"].get("cached_tokens", 0) or 0
                        )
                    if isinstance(raw.get("output_tokens_details"), dict):
                        usage_dict["reasoning_tokens"] = (
                            raw["output_tokens_details"].get("reasoning_tokens", 0) or 0
                        )
                except Exception:
                    log.exception("could not flatten response.usage; skipping")
                    usage_dict = None
            # If this assistant turn had tool calls, we store the
            # FIRST tool call's name + call_id on the turn so the
            # rebuild path can reconstruct the function_call block
            # on the next iteration. The actual full list of tool
            # calls is encoded in the content field.
            primary_call = tool_calls[0] if tool_calls else None
            content_payload: dict = {"text": "\n".join(text_parts)}
            if tool_calls:
                # Embed the full call list so we can replay them
                # exactly. The first call's id is the "anchor" for
                # tool_call_id; the rest are tracked via tool_calls.
                content_payload["tool_calls"] = [
                    {
                        "call_id": c["call_id"],
                        "name": c["name"],
                        "arguments": c["arguments"],
                    }
                    for c in tool_calls
                ]
            # Embed the iteration metrics so the ATIF converter can
            # build a FinalMetrics block without re-parsing the
            # turn log.
            if usage_dict:
                content_payload["metrics"] = usage_dict
            assistant_turn = Turn(
                turn_number=self.turn_log.turn_count + 1,
                role="assistant",
                content=content_payload,
                reasoning=reasoning_text,
                tool_name=primary_call["name"] if primary_call else None,
                tool_call_id=primary_call["call_id"] if primary_call else None,
                usage=usage_dict,
            )
            self.turn_log.append(assistant_turn)

            # Emit events for the ATIF trajectory. Reasoning first
            # (if any), then the assistant text, then one event per
            # tool call. The converter groups consecutive
            # reasoning/assistant_text/tool_call events into a single
            # Step.
            turn_number = assistant_turn.turn_number
            ts = assistant_turn.timestamp
            if reasoning_text:
                self.event_log.append(ReasoningEvent(
                    turn_number=turn_number,
                    timestamp=ts,
                    text=reasoning_text,
                ))
            if text_parts:
                self.event_log.append(AssistantTextEvent(
                    turn_number=turn_number,
                    timestamp=ts,
                    text="\n".join(text_parts),
                    # Embed metrics so the ATIF converter can attribute
                    # them to this assistant turn.
                    metrics=usage_dict,
                ))
            for c in tool_calls:
                # Parse arguments back to a dict for ATIF
                try:
                    args_dict = json.loads(c["arguments"]) if c["arguments"] else {}
                except json.JSONDecodeError:
                    args_dict = {"raw": c["arguments"]}
                self.event_log.append(ToolCallEvent(
                    turn_number=turn_number,
                    timestamp=ts,
                    tool_call_id=c["call_id"],
                    tool_name=c["name"],
                    arguments=args_dict,
                ))

            # No tool calls? We're done.
            if not tool_calls:
                log.info(
                    "agent finished after %d iterations, %d total turns",
                    iteration, self.turn_log.turn_count,
                )
                return "\n".join(text_parts)

            # Dispatch each tool call, append the result, continue.
            for call in tool_calls:
                result_text = self._dispatch_tool(call, reasoning_text)
                # Cap output
                if len(result_text) > self.config.max_tool_output_chars:
                    result_text = (
                        result_text[:self.config.max_tool_output_chars]
                        + f"\n\n[output truncated at {self.config.max_tool_output_chars:,} chars]"
                    )
                self.turn_log.append(Turn(
                    turn_number=self.turn_log.turn_count + 1,
                    role="tool",
                    content={"text": result_text},
                    tool_name=call["name"],
                    tool_call_id=call["call_id"],
                ))
                # Emit a ToolResultEvent for the ATIF trajectory.
                # If the dispatcher captured a subagent trajectory
                # (e.g. from the gardener), attach it to the event so
                # the converter can emit a SubagentTrajectoryRef.
                sub_traj = getattr(self, "_last_subagent_trajectory", None)
                is_subagent = sub_traj is not None and call["name"] == "ask_gardener"
                self.event_log.append(ToolResultEvent(
                    turn_number=self.turn_log.turn_count,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    source_call_id=call["call_id"],
                    tool_name=call["name"],
                    content=result_text,
                    is_subagent=is_subagent,
                    subagent_trajectory=sub_traj,
                ))
                # Clear the captured trajectory so the next tool call
                # starts clean.
                self._last_subagent_trajectory = None

        # If we hit the iteration cap, surface that.
        log.error("agent hit max_tool_iterations=%d, returning partial answer",
                  self.config.max_tool_iterations)
        return (
            f"[agent hit the iteration cap of {self.config.max_tool_iterations}; "
            "the task may not be complete. Check the turn log for context.]"
        )

    # ---- tool dispatch ----------------------------------------------- #

    def _dispatch_tool(self, call: dict, reasoning: str | None) -> str:
        """Dispatch a single tool call. Returns the string result to
        feed back to the model.
        """
        name = call["name"]
        call_id = call["call_id"]
        # Parse arguments (always a JSON string per Responses API).
        try:
            args = json.loads(call["arguments"]) if call["arguments"] else {}
        except json.JSONDecodeError:
            args = {}

        # Troll toll pre-tool-call check
        if self.troll is not None:
            hit = self.troll.check(
                reasoning=reasoning or "",
                tool_name=name,
                tool_input=args,
            )
            if hit is not None:
                # Log the intervention and return the lesson as the
                # tool result. The agent sees the troll's lesson and
                # can either rewrite, defend, or escalate.
                log.warning(
                    "TROLL TOLL: catch=%s pattern=%r tool=%s",
                    hit.kind, hit.pattern, name,
                )
                return hit.to_prompt()

        # Look up the tool in the registry.
        tool = next((t for t in self.tools if t["name"] == name), None)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available: {[t['name'] for t in self.tools]}"

        try:
            ctx = ToolContext(
                cwd=Path(os.getcwd()),
                turn_log=self.turn_log,
                agent=self,
            )
            return tool["fn"](ctx, **args)
        except TypeError as e:
            return f"Error: tool {name!r} called with wrong arguments: {e}"
        except Exception as e:
            log.exception("tool %s failed", name)
            return f"Error: tool {name!r} failed: {e}"

    # ---- API tool list ----------------------------------------------- #

    def _build_api_tools(self) -> list[dict]:
        """Build the Responses API tool list.

        We register the local tools as ``function`` (Codex-native
        type) and the built-in web search as ``web_search``. The API
        figures out the right Codex-native type from the shape.
        """
        api_tools: list[dict] = []
        for t in self.tools:
            api_tools.append({
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            })
        if self.config.enable_web_search:
            api_tools.append({"type": "web_search"})
        if self.config.enable_code_execution:
            api_tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
        if self.config.enable_file_search:
            # Harbor tasks usually have a single working dir; the
            # file_search tool indexes it. Caller configures vector_store.
            pass
        return api_tools

    # ---- the traveling gardener tool --------------------------------- #

    def _make_ask_gardener_tool(self) -> dict:
        """Build the ask_gardener tool. Calls M3 with a slice of the
        raw turn log as its context.
        """
        def _ask_gardener(ctx, query: str, lookback_turns: int = 50) -> str:
            """Ask the traveling gardener (M3, 1M context) a question.

            The gardener sees the last ``lookback_turns`` turns of the
            raw turn log as its only context. It has no other tools
            — its job is to read history and answer.

            Use this when:
              - You need to remember a decision from earlier in the task.
              - You need to recall a tool result that fell out of your
                current attention window.
              - You want a second opinion from a different model.
            Do NOT use this for:
              - The user's CURRENT request (that's your job).
              - Anything that needs the filesystem / shell / web
                (the gardener is read-only on the turn log).

            Args:
                query: The question. Phrase it as a real question.
                lookback_turns: How far back to look. Default 50.
            """
            if not self.config.gardener_enabled or self.gardener_http is None:
                return "Error: gardener is disabled (set MINIMAX_SUBSCRIBER_KEY or unset MIANMI_HEADLESS_GARDENER=0)."
            if not self.config.minimax_api_key:
                return "Error: MINIMAX_SUBSCRIBER_KEY is not set."

            # Pull the last N turns from the log.
            all_turns = list(self.turn_log.iter_turns())
            slice_ = all_turns[-lookback_turns:] if len(all_turns) > lookback_turns else all_turns

            # Build the gardener's messages. We use a system prompt that
            # frames it as a Time Machine sub-agent + a user message
            # with the question + the raw turn slice as context.
            history_blocks: list[dict] = []
            for t in slice_:
                if t.role == "system":
                    continue
                if t.role == "user":
                    history_blocks.append({
                        "role": "user",
                        "content": [{"type": "text", "type": "text", "text": _text(t.content)}],
                    })
                elif t.role == "assistant":
                    text = _text(t.content)
                    if text:
                        history_blocks.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": text}],
                        })
                elif t.role == "tool":
                    # Tool results as plain text in the conversation
                    history_blocks.append({
                        "role": "user",
                        "content": [{"type": "text", "text": f"[tool result for {t.tool_name}]: {_text(t.content)}"}],
                    })

            # The actual user message = the question.
            history_blocks.append({
                "role": "user",
                "content": [{"type": "text", "text": query}],
            })

            # Call M3.
            payload = {
                "model": self.config.gardener_model,
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a Time Machine sub-agent inside mianmi. You have been "
                            "given a raw slice of historical conversation context from a "
                            "previous session. Answer the user's question based solely on "
                            "what you can see in this context. Be specific. Cite turn "
                            "numbers when referencing decisions, code, or reasoning.\n\n"
                            f"Context: last {len(slice_)} turns from the raw turn log."
                        ),
                    },
                    *history_blocks,
                ],
            }
            try:
                resp = self.gardener_http.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                # Capture a sub-agent trajectory for the ATIF converter.
                # The gardener ran in isolation: one system message,
                # one user message (the question + history), one
                # assistant message (the answer). We emit a minimal
                # but valid ATIF trajectory so harbor can render it.
                self._last_subagent_trajectory = {
                    "schema_version": "ATIF-v1.5",
                    "session_id": self.session_id,
                    "trajectory_id": str(uuid.uuid4()),
                    "agent": {
                        "name": "mianmi-gardener",
                        "version": "0.1.0",
                        "model_name": self.config.gardener_model,
                    },
                    "steps": [
                        {
                            "step_id": 1,
                            "timestamp": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                            "source": "user",
                            "message": f"[gardener context: {len(slice_)} turns] {query}",
                        },
                        {
                            "step_id": 2,
                            "timestamp": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                            "source": "agent",
                            "message": answer,
                            "model_name": self.config.gardener_model,
                            "llm_call_count": 1,
                        },
                    ],
                    "final_metrics": {"total_steps": 2},
                }
                return answer
            except httpx.HTTPError as e:
                return f"Error: gardener call failed: {e}"
            except Exception as e:
                return f"Error: gardener parse failed: {e}"

        return {
            "name": "ask_gardener",
            "description": _ask_gardener.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question. Phrase it as a real question.",
                    },
                    "lookback_turns": {
                        "type": "integer",
                        "description": "How far back to look. Default 50.",
                        "default": 50,
                    },
                },
                "required": ["query"],
            },
            "fn": _ask_gardener,
        }

    # ---- lifecycle ---------------------------------------------------- #

    def close(self) -> None:
        self.turn_log.close()
        self.event_log.close()
        if self.gardener_http is not None:
            self.gardener_http.close()

    # ---- trajectory emission ---------------------------------------- #

    def emit_trajectory(
        self,
        path: str | Path | None = None,
        *,
        tool_definitions: list[dict] | None = None,
    ) -> dict:
        """Build an ATIF-v1.5 trajectory from the event log.

        The trajectory is the structured record of everything the
        agent did — designed for harbor's verifier to read after
        a trial. By default it's written to ``<turn_log_dir>/trajectory.json``.

        Returns the dict (also useful for tests).
        """
        return emit_trajectory_from_event_log(
            event_log_path=self.event_log.path,
            output_path=path or self.config.turn_log_path.with_name("trajectory.json"),
            main_model=self.config.main_model,
            reasoning_effort=self.config.reasoning_effort,
            session_id=self.session_id,
            tool_definitions=tool_definitions,
        )


def emit_trajectory_from_event_log(
    *,
    event_log_path: Path,
    output_path: Path,
    main_model: str = "mianmi-headless",
    reasoning_effort: str | None = None,
    session_id: str | None = None,
    tool_definitions: list[dict] | None = None,
) -> dict:
    """Build an ATIF trajectory from an event log file on disk.

    Standalone function — no agent construction needed. Used by
    the CLI's ``emit-trajectory`` subcommand and by ``HeadlessAgent.emit_trajectory``.
    """
    from mianmi_headless.events import EventLog
    log = EventLog(event_log_path)
    try:
        events = list(log.iter_events())
    finally:
        log.close()
    traj = events_to_trajectory(
        events,
        agent_name="mianmi-headless",
        agent_version="0.1.0",
        model_name=main_model,
        tool_definitions=tool_definitions,
        reasoning_effort=reasoning_effort,
        session_id=session_id,
    )
    errors = validate_against_atif(traj)
    if errors:
        log_msg = logging.getLogger("mianmi_headless.atif")
        log_msg.warning("ATIF trajectory has %d validation errors: %s", len(errors), errors)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2, default=str)
    logger = logging.getLogger("mianmi_headless.atif")
    logger.info("wrote ATIF trajectory to %s (%d steps)", output_path, len(traj.get("steps", [])))
    return traj
