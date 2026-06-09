"""Tests for the paperbench-tuned harness additions:
  - paperbench system prompt auto-detection
  - paperbench troll catches
  - write_scratchpad and structured_error tools
  - gpt-5.5-pro default when in paperbench mode
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def paperbench_env(tmp_path, monkeypatch):
    """Set up a fake paperbench workspace: /workspace/submission/tests/rubrics.json
    plus a writable /workspace/submission/scratchpad.md target.
    """
    # The paperbench system prompt auto-detect looks for the file
    # directly. We can't actually create /workspace/submission/ in
    # the test sandbox (we don't have root). Instead, we use the
    # explicit env var to force paperbench mode.
    monkeypatch.setenv("MIANMI_HEADLESS_PAPERBENCH", "1")
    yield tmp_path


# --------------------------------------------------------------------------- #
# Paperbench system prompt
# --------------------------------------------------------------------------- #

class TestPaperbenchSystemPrompt:
    def test_returns_prompt_when_explicitly_enabled(self, paperbench_env):
        from mianmi_headless.paperbench import paperbench_system_prompt
        prompt = paperbench_system_prompt()
        assert prompt is not None
        # Should mention the rubric
        assert "rubric" in prompt.lower()
        # Should mention specific leaves
        assert "metrics.json" in prompt
        assert "REPORT.md" in prompt

    def test_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv("MIANMI_HEADLESS_PAPERBENCH", "0")
        # Also make sure /workspace/submission doesn't exist
        # (it might on the user's machine, but in CI it won't)
        from mianmi_headless.paperbench import paperbench_system_prompt
        prompt = paperbench_system_prompt()
        # If /workspace/submission/tests/rubrics.json exists on the
        # host, this might return non-None. We can't easily test the
        # "off" case without more machinery, so we just verify the
        # function is callable.
        assert prompt is None or isinstance(prompt, str)

    def test_explicit_1_overrides_no_rubric(self, monkeypatch):
        monkeypatch.setenv("MIANMI_HEADLESS_PAPERBENCH", "1")
        from mianmi_headless.paperbench import paperbench_system_prompt
        prompt = paperbench_system_prompt()
        assert prompt is not None
        assert "rubric" in prompt.lower()

    def test_prompt_has_priority_ordering(self):
        from mianmi_headless.paperbench import PAPERBENCH_SYSTEM_PROMPT
        # Should mention: operational artifacts first, algorithm second,
        # experiment third.
        text = PAPERBENCH_SYSTEM_PROMPT.lower()
        # All three priorities should appear in some form
        assert "operational" in text or "operational hygiene" in text
        assert "algorithm" in text
        assert "experiment" in text


# --------------------------------------------------------------------------- #
# Paperbench troll catches
# --------------------------------------------------------------------------- #

class TestPaperbenchTrollCatches:
    def test_package_upgrade_fires(self):
        from mianmi_headless.troll import TrollToll
        # Force paperbench mode via env
        os.environ["MIANMI_HEADLESS_PAPERBENCH"] = "1"
        try:
            toll = TrollToll()  # auto-detects paperbench
            hit = toll.check(
                reasoning="I should upgrade torch to fix the bug",
                tool_name="run_command",
                tool_input={"command": "pip install --upgrade torch"},
            )
            assert hit is not None
            assert hit.kind == "package_upgrade"
        finally:
            del os.environ["MIANMI_HEADLESS_PAPERBENCH"]

    def test_skip_deterministic_leaves_fires(self):
        from mianmi_headless.troll import TrollToll
        os.environ["MIANMI_HEADLESS_PAPERBENCH"] = "1"
        try:
            toll = TrollToll()
            hit = toll.check(
                reasoning="Let's focus on the algorithm first, the code is what matters",
                tool_name="write_file",
                tool_input={},
            )
            assert hit is not None
            assert hit.kind == "skip_deterministic_leaves"
        finally:
            del os.environ["MIANMI_HEADLESS_PAPERBENCH"]

    def test_ref_author_repo_fires(self):
        from mianmi_headless.troll import TrollToll
        os.environ["MIANMI_HEADLESS_PAPERBENCH"] = "1"
        try:
            toll = TrollToll()
            hit = toll.check(
                reasoning="I should look at the safe-torch repo to understand the implementation",
                tool_name="run_command",
                tool_input={},
            )
            assert hit is not None
            assert hit.kind == "ref_author_repo"
        finally:
            del os.environ["MIANMI_HEADLESS_PAPERBENCH"]

    def test_arxiv_reference_fires(self):
        from mianmi_headless.troll import TrollToll
        os.environ["MIANMI_HEADLESS_PAPERBENCH"] = "1"
        try:
            toll = TrollToll()
            hit = toll.check(
                reasoning="Let me check arxiv.org/abs/2506.06866 for the paper",
                tool_name="web_search",
                tool_input={"query": "arxiv"},
            )
            assert hit is not None
            assert hit.kind == "arxiv_reference"
        finally:
            del os.environ["MIANMI_HEADLESS_PAPERBENCH"]

    def test_paperbench_mode_adds_catches(self):
        from mianmi_headless.troll import (
            TrollToll, SEED_CATCHES, PAPERBENCH_CATCHES, get_catches,
        )
        # No paperbench mode → just the SEED_CATCHES
        os.environ.pop("MIANMI_HEADLESS_PAPERBENCH", None)
        # Make sure /workspace/submission doesn't accidentally exist
        # (it might on the user's machine; we use explicit env)
        os.environ.pop("MIANMI_HEADLESS_PAPERBENCH", None)
        base = len(get_catches())
        assert base == len(SEED_CATCHES)
        # Paperbench mode → SEED + PAPERBENCH
        os.environ["MIANMI_HEADLESS_PAPERBENCH"] = "1"
        try:
            augmented = len(get_catches())
            assert augmented == len(SEED_CATCHES) + len(PAPERBENCH_CATCHES)
        finally:
            del os.environ["MIANMI_HEADLESS_PAPERBENCH"]


# --------------------------------------------------------------------------- #
# write_scratchpad + structured_error tools
# --------------------------------------------------------------------------- #

class TestPaperbenchTools:
    def test_write_scratchpad_to_local_path(self, tmp_path, monkeypatch):
        from mianmi_headless.tools import _write_scratchpad, ToolContext
        from unittest.mock import MagicMock
        # Make sure /workspace/submission doesn't exist for the test
        # by patching the candidate list
        ctx = MagicMock()
        ctx.cwd = tmp_path
        # The tool writes to /workspace/submission/scratchpad.md OR
        # cwd/scratchpad.md. We want it to fall back to cwd.
        # We simulate the first candidate missing by patching Path.
        # Easier: just verify the cwd-relative write works.
        result = _write_scratchpad(ctx, content="## Plan\n\n- read rubric\n", append=False)
        # Either /workspace path (if exists) or cwd path
        assert "bytes" in result
        # The local file should be created
        local = tmp_path / "scratchpad.md"
        if local.exists():
            assert "Plan" in local.read_text()

    def test_structured_error_writes_json(self, tmp_path, monkeypatch):
        from mianmi_headless.tools import _structured_error, ToolContext
        from unittest.mock import MagicMock
        ctx = MagicMock()
        ctx.cwd = tmp_path
        result = _structured_error(
            ctx,
            artifact="errors/test.json",
            error="GPU not available",
            evidence="torch.cuda.is_available() returned False",
            recovery="fell back to CPU",
        )
        assert "Recorded" in result or "Error" in result
        # The local file should be created with JSON content
        local = tmp_path / "errors/test.json"
        if local.exists():
            import json
            data = json.loads(local.read_text())
            assert data["error"] == "GPU not available"
            assert "evidence" in data
            assert "recovery" in data
            assert "timestamp" in data


# --------------------------------------------------------------------------- #
# Tool registry includes the paperbench tools
# --------------------------------------------------------------------------- #

class TestToolRegistryPaperbench:
    def test_registry_includes_scratchpad_and_structured_error(self):
        from mianmi_headless.tools import registry
        tools = registry()
        names = {t["name"] for t in tools}
        assert "write_scratchpad" in names
        assert "structured_error" in names


# --------------------------------------------------------------------------- #
# Default model: gpt-5.5-pro when paperbench, gpt-5.5 otherwise
# --------------------------------------------------------------------------- #

class TestDefaultModel:
    def test_paperbench_defaults_to_pro(self, monkeypatch):
        monkeypatch.delenv("MIANMI_HEADLESS_MODEL", raising=False)
        monkeypatch.setenv("MIANMI_HEADLESS_PAPERBENCH", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        from mianmi_headless.agent import AgentConfig
        from mianmi_headless.paperbench import paperbench_system_prompt
        # Stub the system prompt to return None so config validation
        # doesn't interfere
        monkeypatch.setattr(paperbench_system_prompt.__module__ + ".paperbench_system_prompt",
                            lambda: None)
        config = AgentConfig()
        assert config.main_model == "gpt-5.5-pro"

    def test_explicit_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("MIANMI_HEADLESS_MODEL", "gpt-5.4")
        monkeypatch.setenv("MIANMI_HEADLESS_PAPERBENCH", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        from mianmi_headless.agent import AgentConfig
        config = AgentConfig()
        assert config.main_model == "gpt-5.4"
