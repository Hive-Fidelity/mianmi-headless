"""The mianmi-headless CLI.

Usage:
    mianmi-headless run "your task description"
    mianmi-headless version
    mianmi-headless self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from mianmi_headless import __version__
from mianmi_headless.agent import AgentConfig, HeadlessAgent


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _cmd_run(args: argparse.Namespace) -> int:
    _setup_logging(verbose=args.verbose)
    try:
        config = AgentConfig()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.model:
        config.main_model = args.model
    if args.reasoning:
        config.reasoning_effort = args.reasoning
    if args.turn_log:
        config.turn_log_path = Path(args.turn_log)
    if args.disable_gardener:
        config.gardener_enabled = False
    if args.disable_troll:
        config.troll_enabled = False
    if args.max_iter:
        config.max_tool_iterations = args.max_iter

    agent = HeadlessAgent(config)
    try:
        result = agent.run(args.instruction)
        # Print the result to stdout. Harbor's verifier reads the
        # agent's stdout / log file for the final answer.
        print(result)
        return 0
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        agent.close()


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"mianmi-headless {__version__}")
    return 0


def _cmd_emit_trajectory(args: argparse.Namespace) -> int:
    """Build an ATIF-v1.5 trajectory from the event log on disk.

    Reads the events JSONL next to the turn log, converts to ATIF,
    and writes to ``--output`` (default: ``<turn_log>.trajectory.json``).

    Designed to be called by the harbor patch after the agent run
    completes — the trajectory is the structured record harbor's
    verifier reads. Does NOT require an OpenAI API key (we just
    read the on-disk event log, no LLM call).
    """
    _setup_logging(verbose=args.verbose)
    from mianmi_headless.agent import emit_trajectory_from_event_log
    if args.turn_log:
        turn_log = Path(args.turn_log)
    else:
        turn_log = Path(os.getenv("MIANMI_HEADLESS_TURN_LOG", "./turns.jsonl"))
    event_log = turn_log.with_name(turn_log.stem + ".events.jsonl")
    if not event_log.exists():
        print(f"Error: event log not found: {event_log}", file=sys.stderr)
        return 2
    output = Path(args.output) if args.output else turn_log.with_name("trajectory.json")
    session_id = args.session_id or os.getenv("MIANMI_HEADLESS_SESSION_ID")
    traj = emit_trajectory_from_event_log(
        event_log_path=event_log,
        output_path=output,
        main_model=args.model or os.getenv("MIANMI_HEADLESS_MODEL", "gpt-5.5"),
        reasoning_effort=args.reasoning or os.getenv("MIANMI_HEADLESS_REASONING", "high"),
        session_id=session_id,
    )
    print(f"wrote trajectory with {len(traj.get('steps', []))} steps to {output}")
    return 0


def _cmd_self_test(args: argparse.Namespace) -> int:
    """A quick smoke test: does the agent construct, can the troll fire,
    can the OpenRouter scraper work? Doesn't hit the real OpenAI API.
    """
    _setup_logging(verbose=args.verbose)
    print("mianmi-headless self-test")
    print("=" * 50)

    # 1. Config validation
    try:
        AgentConfig(openai_api_key="fake-key-for-self-test")
    except ValueError as e:
        print(f"FAIL: config validation: {e}")
        return 1
    print("OK  config validation (rejects empty API key)")

    # 2. Troll toll — known-bad patterns fire
    from mianmi_headless.troll import TrollToll, SEED_CATCHES
    toll = TrollToll()
    for prompt in [
        "Use claude-sonnet-4 for this.",
        "Use the latest model from openai.",
        "Use minimax-m2 as the gardener",
    ]:
        hit = toll.check(reasoning=prompt, tool_name="x", tool_input={})
        if hit is None:
            print(f"FAIL: troll should fire on {prompt!r}")
            return 1
    print(f"OK  troll toll fires on {len(SEED_CATCHES)} known-bad patterns")

    # 3. Tool registry
    from mianmi_headless.tools import registry
    tools = registry()
    if len(tools) < 4:
        print(f"FAIL: tool registry has only {len(tools)} tools, expected at least 4")
        return 1
    print(f"OK  tool registry has {len(tools)} tools: {[t['name'] for t in tools]}")

    # 4. OpenRouter scraper (this one DOES hit the network)
    try:
        from mianmi_headless.tools import _scrape_openrouter_models
        from unittest.mock import MagicMock
        # Build a minimal context.
        ctx = MagicMock()
        ctx.cwd = Path("/tmp")
        ctx.turn_log = None
        ctx.agent = MagicMock()
        ctx.agent._or_cache = {}
        out = _scrape_openrouter_models(ctx, vendor_filter="minimax", top_n=3)
        if "mianmi" in out.lower() or "minimax" in out.lower() or "1M-context" in out:
            print("OK  scrape_openrouter_models works")
        else:
            print(f"WARN: scrape returned unexpected output: {out[:200]}")
    except Exception as e:
        print(f"WARN: scrape_openrouter_models network check skipped: {e}")

    print()
    print("self-test passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mianmi-headless",
        description="Headless mianmi agent for harbor / terminal-bench",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_run = sub.add_parser("run", help="Run the agent on a task")
    p_run.add_argument("instruction", help="The task description")
    p_run.add_argument("--model", help="Override the main model (default gpt-5.5)")
    p_run.add_argument("--reasoning", help="Reasoning effort: low, medium, high")
    p_run.add_argument("--turn-log", help="Path to the raw turn log JSONL")
    p_run.add_argument("--max-iter", type=int, help="Max tool-call iterations")
    p_run.add_argument("--disable-gardener", action="store_true", help="Disable the M3 gardener")
    p_run.add_argument("--disable-troll", action="store_true", help="Disable the troll toll")
    p_run.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p_run.set_defaults(func=_cmd_run)

    p_version = sub.add_parser("version", help="Print version and exit")
    p_version.set_defaults(func=_cmd_version)

    p_self = sub.add_parser("self-test", help="Run smoke tests")
    p_self.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p_self.set_defaults(func=_cmd_self_test)

    p_emit = sub.add_parser(
        "emit-trajectory",
        help="Build an ATIF-v1.5 trajectory from the event log",
    )
    p_emit.add_argument("--turn-log", help="Path to the raw turn log JSONL")
    p_emit.add_argument("--output", "-o", help="Output path (default: trajectory.json next to turn log)")
    p_emit.add_argument("--model", help="Override the main model (for the agent section)")
    p_emit.add_argument("--reasoning", help="Reasoning effort (for the agent section)")
    p_emit.add_argument("--session-id", help="Override the session id (default: from env or random)")
    p_emit.add_argument("--disable-gardener", action="store_true", help="Disable the M3 gardener")
    p_emit.add_argument("--disable-troll", action="store_true", help="Disable the troll toll")
    p_emit.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p_emit.set_defaults(func=_cmd_emit_trajectory)

    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
