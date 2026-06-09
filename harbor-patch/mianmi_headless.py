"""mianmi-headless — the harbor agent patch.

This is the file you drop into your harbor fork at:

    src/harbor/agents/installed/mianmi_headless.py

Then add ``MIANMI_HEADLESS = "mianmi-headless"`` to your
``AgentName`` enum (in ``src/harbor/models/agent/name.py``) and
register it in ``registry.json`` if you want to use it via the
name string.

The patch is intentionally small — the heavy lifting happens in the
``mianmi-headless`` package itself (which harbor installs into the
container via ``pip install -e .``). This file just glues the
harbor trial runner to the package's CLI.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any, Literal

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    EnvVar,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.trajectory_utils import format_trajectory_json


class MianmiHeadless(BaseInstalledAgent):
    """The mianmi-headless agent.

    Drops the package into the container (via pip), then runs the
    CLI on the task instruction. The CLI is ``mianmi-headless run``
    — see the package's CLI module for the full surface.
    """

    SUPPORTS_ATIF: bool = False  # The CLI doesn't emit ATIF natively yet.
    _OUTPUT_FILENAME = "mianmi_headless_output.txt"

    # CLI flags that map to environment variables on the agent process.
    # Harbor wires these via EnvVar descriptors so users can set them
    # at the trial level without subclassing.
    ENV_VARS = [
        EnvVar(
            "main_model",
            env="MIANMI_HEADLESS_MODEL",
            type="str",
            default="gpt-5.5",
        ),
        EnvVar(
            "reasoning_effort",
            env="MIANMI_HEADLESS_REASONING",
            type="enum",
            choices=["low", "medium", "high"],
            default="high",
        ),
        EnvVar(
            "max_iter",
            env="MIANMI_HEADLESS_MAX_ITER",
            type="int",
            default=500,
        ),
        EnvVar(
            "enable_gardener",
            env="MIANMI_HEADLESS_GARDENER",
            type="enum",
            choices=["0", "1"],
            default="1",
        ),
        EnvVar(
            "enable_troll",
            env="MIANMI_HEADLESS_TROLL",
            type="enum",
            choices=["0", "1"],
            default="1",
        ),
    ]

    @staticmethod
    def name() -> str:
        # Add this to your AgentName enum (see top of file).
        return AgentName.MIANMI_HEADLESS.value

    @property
    def _trajectory_path(self) -> Path:
        return EnvironmentPaths.agent_dir / "trajectory.json"

    def get_version_command(self) -> str | None:
        return "mianmi-headless version"

    def parse_version(self, stdout: str) -> str:
        # Output of `mianmi-headless version` is "mianmi-headless 0.1.0"
        text = stdout.strip()
        if text.startswith("mianmi-headless"):
            return text.split()[-1]
        return text

    async def install(self, environment: BaseEnvironment) -> None:
        """Install the mianmi-headless package in the container.

        The package source defaults to the local checkout (``MIANMI_HEADLESS_REPO=./mianmi-headless``),
        but you can override with a git URL, a local path, or a PyPI
        version pin.
        """
        # 1. System deps (pip, build tools)
        await self.exec_as_root(
            environment,
            command=(
                "if command -v apt-get &>/dev/null; then"
                "  apt-get update && apt-get install -y python3-pip;"
                " elif command -v yum &>/dev/null; then"
                "  yum install -y python3-pip;"
                " elif command -v apk &>/dev/null; then"
                "  apk add --no-cache py3-pip;"
                " else"
                '  echo "Warning: No known package manager found, assuming pip is available" >&2;'
                " fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

        # 2. Install the package. Default is local source from a sibling
        # directory called ``mianmi-headless``; override via
        # ``MIANMI_HEADLESS_REPO=git+https://...`` to pull from GitHub.
        repo = self._get_env("MIANMI_HEADLESS_REPO") or "./mianmi-headless"
        version = self._get_env("MIANMI_HEADLESS_VERSION") or ""
        install_cmd = (
            "set -euo pipefail; "
            "python3 -m pip install --upgrade pip wheel setuptools; "
        )
        if version:
            install_cmd += f"python3 -m pip install 'mianmi-headless=={version}'"
        elif repo.startswith("git+") or repo.startswith("http"):
            install_cmd += f"python3 -m pip install '{repo}'"
        else:
            install_cmd += f"python3 -m pip install -e '{repo}'"
        install_cmd += " && mianmi-headless version"
        await self.exec_as_agent(environment, command=install_cmd)

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run the agent on the task instruction.

        The CLI runs the full agent loop and writes:
          - The raw turn log to ``./turns.jsonl`` (in the task working dir)
          - The structured event log to ``./turns.events.jsonl``
          - The final answer to stdout (captured to agent_dir)
          - The ATIF-v1.5 trajectory to ``./trajectory.json`` (post-run)

        harbor's verifier reads ``trajectory.json`` for structured
        metrics; the raw turn log is the source of truth.
        """
        output_path = EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME
        traj_path = EnvironmentPaths.agent_dir / "trajectory.json"

        # Build the command. All EnvVar values are auto-injected by
        # ``exec_as_agent`` via the ``_extra_env`` plumbing.
        max_iter = self._resolved_env_vars.get("MIANMI_HEADLESS_MAX_ITER", "500")
        model = self._resolved_env_vars.get("MIANMI_HEADLESS_MODEL", "gpt-5.5")
        reasoning = self._resolved_env_vars.get("MIANMI_HEADLESS_REASONING", "high")

        # Phase 1: run the agent. stdout → agent_dir output file.
        run_cmd = (
            "mianmi-headless run "
            f"--model {shlex.quote(model)} "
            f"--reasoning {shlex.quote(reasoning)} "
            f"--max-iter {max_iter} "
            f"--turn-log ./turns.jsonl "
            f"--verbose "
            f"{shlex.quote(instruction)} "
            f"2>&1 </dev/null | tee {output_path.as_posix()}"
        )
        await self.exec_as_agent(environment, command=run_cmd)

        # Phase 2: convert the event log to an ATIF trajectory.
        # Best-effort — if it fails (e.g. the agent crashed before
        # writing any events), log and move on. harbor can still
        # use the output file + raw turn log.
        emit_cmd = (
            "mianmi-headless emit-trajectory "
            f"--turn-log ./turns.jsonl "
            f"--output {traj_path.as_posix()} "
            f"--model {shlex.quote(model)} "
            f"2>&1 | tee -a {output_path.as_posix()}"
        )
        try:
            await self.exec_as_agent(environment, command=emit_cmd)
        except Exception:
            # Non-fatal: harbor can still parse the raw turn log.
            self.logger.warning("ATIF trajectory emission failed; continuing")

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Backfill context from the ATIF trajectory.

        The CLI emits ``trajectory.json`` (ATIF-v1.5) via
        ``mianmi-headless emit-trajectory``. We read its
        ``final_metrics`` block to populate the harbor context.
        Falls back to cheap text extraction from the output file
        if the trajectory is missing.
        """
        traj_path = self.logs_dir / "trajectory.json"
        if traj_path.exists():
            try:
                import json
                traj = json.loads(traj_path.read_text(encoding="utf-8"))
            except Exception:
                traj = None
            if traj is not None:
                fm = traj.get("final_metrics") or {}
                if (v := fm.get("total_prompt_tokens")) is not None:
                    context.n_input_tokens = v
                if (v := fm.get("total_cached_tokens")) is not None:
                    context.n_cache_tokens = v
                if (v := fm.get("total_completion_tokens")) is not None:
                    context.n_output_tokens = v
                if (v := fm.get("total_cost_usd")) is not None:
                    context.cost_usd = v
                return

        # Fallback: cheap text extraction from the output file.
        output_path = self.logs_dir / self._OUTPUT_FILENAME
        if not output_path.exists():
            return
        try:
            text = output_path.read_text(encoding="utf-8")
        except Exception:
            return
        n_input = 0
        n_output = 0
        for line in text.splitlines():
            if "input_tokens" in line and ":" in line:
                try:
                    n_input = int(line.split(":")[-1].strip().split()[0].replace(",", ""))
                except Exception:
                    pass
            if "output_tokens" in line and ":" in line:
                try:
                    n_output = int(line.split(":")[-1].strip().split()[0].replace(",", ""))
                except Exception:
                    pass
        context.n_input_tokens = n_input
        context.n_output_tokens = n_output
