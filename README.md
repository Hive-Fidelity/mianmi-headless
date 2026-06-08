# mianmi-headless

**The headless variant of mianmi, designed to run for hours inside a
harbor / terminal-bench container, doing hard ML research and coding
tasks without complaining about dollars, context, or fallbacks.**

* Uses the **OpenAI Responses API** directly (no LiteLLM, no chat-completions fallback).
* **1M context window, hard.** `truncation="disabled"`. If the model
  can't fit, you get an error — never a silent cut.
* **Codex-native tool types** (`function_call`, `custom_tool_call`,
  `web_search_call`, `file_search_call`, `code_execution_call`).
* A **traveling gardener** (M3, 1M context) that lives next to the main
  agent and answers "what was I doing 200 turns ago?" / "what does the
  code look like?" without burning the main agent's context window.
* A **troll toll** that pre-tool-call hooks catch stale model ids and
  phrasings before they reach the API. Cheap fix for "training data
  drift" failures.
* **No Postgres, no FastAPI, no server.** Just a CLI:
  `mianmi-headless run "your instruction here"`.

## What this is NOT

This is not the full mianmi desktop/server product. The full product
lives at [mianmi-agent](https://github.com/kruge/mianmi-agent) and has
the FastAPI server, the conversation manager with sliding window, the
Postgres-backed CD-album architecture, the macOS SwiftUI app, etc. This
repo is the **headless variant** — a single Python package that
provides an optimal harness for `gpt-5.5` on long-running, difficult,
research-based ML benchmarking tasks where context, accuracy, and
predictability matter more than dollars.

## Why no LiteLLM

LiteLLM is great for routing across providers. It is bad for
**guarantees**. When a research benchmark takes 8 hours and silently
truncates your context at 272K to "save you money", you discover
this 6 hours too late. By calling the OpenAI SDK directly, we get:

* `truncation="disabled"` — the API will **refuse** rather than
  truncate. Loud failure is the goal.
* `previous_response_id` — the response-chaining primitive the
  Responses API was designed for. No "we lost 30 turns of context,
  here are 3 random ones instead" middleman.
* `reasoning={"effort": "high"}` — proper reasoning config, not
  whatever the chat-completions API allows.
* Codex-native tool types out of the box.

## Install

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
export MINIMAX_SUBSCRIBER_KEY=...   # for the gardener
mianmi-headless run "your task"
```

## Running inside a harbor container

Harbor's `BaseInstalledAgent.install()` calls `pip install -e .` (or
equivalent) in the container, then `run()` to start the agent. The
harbor-patch/ directory in this repo has the 2-3 file patch that
plugs mianmi-headless into your harbor fork.

## The gardener

The gardener is a sidecar M3 (Minimax-M3, 1M context, subscriber API)
that lives in the same process as the main agent. When the main
agent needs historical context, it calls the `ask_gardener` tool.
The gardener reads the on-disk raw turn log (one JSONL file per
session) and answers with citations to specific turn numbers.

This is a stripped-down version of the full mianmi Time Machine — no
project tab cards, no important conversations index, no snapshot. Just
the raw turn log and the gardener. Good enough for the harbor
benchmark case.

## Troll toll

Same idea as the full mianmi project, simplified. The pre-tool-call
hook scans the agent's reasoning + tool input for known-bad patterns
(stale model ids, "the latest model" phrasing, etc.) and forces the
agent to re-verify. See `mianmi_headless.troll`.
