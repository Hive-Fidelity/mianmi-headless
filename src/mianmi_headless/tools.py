"""The default tool registry for the headless harness.

The full mianmi project has filesystem / shell / ground_truth /
time_machine / gpu_hunt tools, all bound to a workspace. The headless
variant ships with a small, environment-aware subset:

  * ``read_file``    — read a file (relative to cwd or absolute)
  * ``write_file``   — write a file (relative to cwd or absolute)
  * ``run_command``  — execute a shell command, return stdout/stderr
  * ``list_files``   — list files matching a glob, with sizes
  * ``scrape_openrouter_models`` — fetch the live OpenRouter model
    list, the troll's preferred escape hatch for "what model is
    current?" questions

These are intentionally minimal. The harbor benchmark tasks are
typically "given a problem statement, write code, run it, submit
the result" — the agent needs files + shell, not much else.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from mianmi_headless.turn_log import TurnLog

log = logging.getLogger("mianmi_headless.tools")


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass
class ToolContext:
    """Per-call context passed to every tool function.

    Tools that need the turn log or a reference back to the agent
    can ask for it here. Tools that only need the cwd can ignore
    the rest.
    """
    cwd: Path
    turn_log: TurnLog
    agent: Any  # HeadlessAgent — for tools that need to peek at config


# --------------------------------------------------------------------------- #
# Tool functions
# --------------------------------------------------------------------------- #

def _resolve_path(ctx: ToolContext, path: str) -> Path:
    """Resolve a path relative to the cwd, refusing escapes."""
    p = Path(path)
    if not p.is_absolute():
        p = ctx.cwd / p
    p = p.resolve()
    return p


def read_file(ctx: ToolContext, path: str, max_bytes: int = 200_000) -> str:
    """Read a file's contents.

    Args:
        path: Path to the file. Relative to cwd or absolute.
        max_bytes: Hard cap on file size. Files larger than this
            are truncated.
    """
    p = _resolve_path(ctx, path)
    if not p.is_file():
        return f"Error: not a file: {p}"
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error: {e}"
    if len(content) > max_bytes:
        content = content[:max_bytes] + f"\n\n[truncated at {max_bytes:,} bytes]"
    return content


def write_file(ctx: ToolContext, path: str, content: str) -> str:
    """Write ``content`` to ``path``, creating parent dirs as needed.

    Args:
        path: Path to the file. Relative to cwd or absolute.
        content: The full file contents.
    """
    p = _resolve_path(ctx, path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as e:
        return f"Error: {e}"
    return f"Wrote {len(content):,} bytes to {p}"


def run_command(
    ctx: ToolContext,
    command: str,
    timeout_sec: int = 600,
) -> str:
    """Execute a shell command and return its combined output.

    Args:
        command: The shell command to run. Use bash.
        timeout_sec: Max wall-clock time. Default 10 minutes.
    """
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(ctx.cwd),
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout_sec}s"
    except Exception as e:
        return f"Error: {e}"
    out = result.stdout
    err = result.stderr
    rc = result.returncode
    parts = []
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    parts.append(f"\n[exit code: {rc}]")
    return "\n".join(parts)


def list_files(ctx: ToolContext, pattern: str = "*", limit: int = 200) -> str:
    """List files matching a glob, with sizes and mtimes.

    Args:
        pattern: Glob pattern (relative to cwd). Default ``*``.
        limit: Max number of entries to return.
    """
    base = ctx.cwd
    try:
        matches = sorted(base.glob(pattern))
    except Exception as e:
        return f"Error: {e}"
    if not matches:
        return f"No files matching {pattern!r} in {base}."
    lines = [f"Files matching {pattern!r} (showing up to {limit}):"]
    for m in matches[:limit]:
        try:
            st = m.stat()
            size_kb = st.st_size / 1024
            lines.append(f"  {str(m.relative_to(base)):60s} {size_kb:8.1f} KB")
        except Exception:
            lines.append(f"  {m}  (stat failed)")
    if len(matches) > limit:
        lines.append(f"  ... and {len(matches) - limit} more")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The OpenRouter scraper — the troll's preferred up-to-date source
# --------------------------------------------------------------------------- #

def _scrape_openrouter_models(
    ctx: ToolContext,
    vendor_filter: str | None = None,
    top_n: int = 30,
    force_refresh: bool = False,
) -> str:
    """Fetch the live OpenRouter model list and return a short summary.

    The recommended escape hatch for any ``model_id_stale`` or
    ``phrasing_stale`` troll toll catch.

    Args:
        vendor_filter: Optional case-insensitive substring filter on
            the model id (e.g. ``"minimax"``, ``"openai"``).
        top_n: How many models to show. Default 30.
        force_refresh: Bypass the in-process cache.
    """
    # Tiny in-process cache. The endpoint is fast but we don't need
    # to hit it on every troll resolution.
    cache_key = (vendor_filter, top_n)
    cache = getattr(ctx.agent, "_or_cache", None)
    if cache is None:
        cache = {}
        ctx.agent._or_cache = cache
    now = __import__("time").monotonic()
    cached = cache.get(cache_key)
    if not force_refresh and cached and now - cached["fetched_at"] < 600:
        return cached["text"]

    try:
        req = Request(
            OPENROUTER_MODELS_URL,
            headers={"User-Agent": "mianmi-headless/0.1 (troll-toll escape hatch)"},
        )
        with urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        return f"OpenRouter scrape failed: {e}"

    models = payload.get("data", [])
    if vendor_filter:
        v = vendor_filter.lower()
        models = [m for m in models if v in (m.get("id", "")).lower()]

    # 1M-context candidates first
    big_ctx = [
        m for m in models
        if ((m.get("top_provider") or {}).get("context_length") or 0) >= 1_000_000
    ]
    lines = [f"OpenRouter models (showing {min(top_n, len(models))} of {len(models)} total"
             f"{', vendor=' + vendor_filter if vendor_filter else ''}):"]
    if big_ctx:
        lines.append("\n1M-context candidates (hard requirement for the gardener):")
        for m in big_ctx[:5]:
            ctx_len = (m.get("top_provider") or {}).get("context_length")
            lines.append(f"  - {m.get('id')!r:60s}  ctx={ctx_len:,}")
    lines.append("\nTop of the full list:")
    for m in models[:top_n]:
        ctx_len = (m.get("top_provider") or {}).get("context_length")
        ctx_str = f"{ctx_len:,}" if ctx_len else "?"
        lines.append(f"  - {m.get('id')!r:60s}  ctx={ctx_str:>10s}")
    text = "\n".join(lines)
    cache[cache_key] = {"text": text, "fetched_at": now}
    return text


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def registry() -> list[dict]:
    """Return the list of tools the agent has access to.

    Each entry has ``name``, ``description``, ``parameters`` (JSON
    Schema), and ``fn`` (the callable). The agent registers these
    as ``function``-type tools on the Responses API.
    """
    return [
        {
            "name": "read_file",
            "description": read_file.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file. Relative to cwd or absolute."},
                    "max_bytes": {"type": "integer", "default": 200_000, "description": "Hard cap on file size."},
                },
                "required": ["path"],
            },
            "fn": read_file,
        },
        {
            "name": "write_file",
            "description": write_file.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file. Relative to cwd or absolute."},
                    "content": {"type": "string", "description": "The full file contents."},
                },
                "required": ["path", "content"],
            },
            "fn": write_file,
        },
        {
            "name": "run_command",
            "description": run_command.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command. Use bash."},
                    "timeout_sec": {"type": "integer", "default": 600, "description": "Max wall-clock time."},
                },
                "required": ["command"],
            },
            "fn": run_command,
        },
        {
            "name": "list_files",
            "description": list_files.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "default": "*", "description": "Glob pattern."},
                    "limit": {"type": "integer", "default": 200, "description": "Max entries."},
                },
            },
            "fn": list_files,
        },
        {
            "name": "scrape_openrouter_models",
            "description": _scrape_openrouter_models.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor_filter": {"type": "string", "description": "Optional substring filter on model id."},
                    "top_n": {"type": "integer", "default": 30, "description": "Max models to show."},
                    "force_refresh": {"type": "boolean", "default": False, "description": "Bypass cache."},
                },
            },
            "fn": _scrape_openrouter_models,
        },
    ]
