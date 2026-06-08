# Harbor integration patch for mianmi-headless

Drop these two files into your harbor fork and add the enum entry.

## Files to add

### `src/harbor/agents/installed/mianmi_headless.py`

Copy from `harbor-patch/mianmi_headless.py` in this repo.

### `src/harbor/models/agent/name.py` — add the enum entry

```python
class AgentName(str, Enum):
    # ... existing entries ...
    MIANMI_HEADLESS = "mianmi-headless"
```

(If your `AgentName` enum is a plain string constant, just add
`MIANMI_HEADLESS = "mianmi-headless"` next to the others.)

## How harbor calls it

1. The trial runner resolves the agent name (`mianmi-headless`) to
   the `MianmiHeadless` class via the import path
   `harbor.agents.installed.mianmi_headless:MianmiHeadless`.
2. `install(environment)` runs `install.sh` in the container, which
   `pip install -e .`s the `mianmi-headless` package from
   `MIANMI_HEADLESS_REPO` (default: a sibling `./mianmi-headless`
   directory, which harbor will have mounted).
3. `run(instruction, environment, context)` invokes the CLI:
   `mianmi-headless run --model gpt-5.5 --reasoning high --max-iter 500 <instruction>`.
   The CLI writes:
   - The final answer to `mianmi_headless_output.txt` (in the agent dir).
   - The full raw turn log to `./turns.jsonl` (in the task working dir).
4. `populate_context_post_run` extracts token counts from the output
   for the harbor metrics.

## Env vars the trial can set

| Env var                          | Default | Effect |
|----------------------------------|---------|--------|
| `OPENAI_API_KEY`                 | (required) | Front model auth |
| `MINIMAX_SUBSCRIBER_KEY`         | (optional) | Gardener (M3) auth |
| `MIANMI_HEADLESS_MODEL`          | `gpt-5.5` | Override the main model |
| `MIANMI_HEADLESS_REASONING`      | `high` | Reasoning effort |
| `MIANMI_HEADLESS_MAX_ITER`       | `500` | Max tool-call iterations per turn |
| `MIANMI_HEADLESS_GARDENER`       | `1` | Set `0` to disable the M3 sidecar |
| `MIANMI_HEADLESS_TROLL`          | `1` | Set `0` to disable the troll toll |
| `MIANMI_HEADLESS_REPO`           | `./mianmi-headless` | Override the install source |
| `MIANMI_HEADLESS_VERSION`        | (none) | Pin a specific version |

## Why this is small

The mianmi-headless package does all the heavy lifting (the agent
loop, the gardener, the troll toll, the OpenAI Responses API call).
The harbor file is just a glue layer that:

- Resolves env vars
- Runs `pip install` in the container
- Invokes the CLI with the right flags
- Captures the output for the verifier

The full agent logic is in the package itself, so the same harness
can be used outside harbor (just `mianmi-headless run "your task"`).
