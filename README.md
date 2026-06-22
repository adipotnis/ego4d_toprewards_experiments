# Ego4D TOPReward Experiments

Standalone re-implementation of [TOPReward](https://topreward.github.io/webpage/) ("Token Probabilities as Hidden Zero-Shot Rewards") run on Ego4D first-person manipulation clips.

Self-contained: does not import from the upstream `topreward` Python package. Has its own `pyproject.toml` / `uv.lock`, so the experiments dir can stand on its own with `uv sync`. (The cluster sbatch happens to reuse the upstream's `.venv_tr/` because it's already provisioned — see [External paths](#external-paths) — but that's an optimization, not a requirement.)

## What it does

For every Ego4D clip (egocentric video + narration caption) the script computes:

- **Logits reward** — `log P(caption + " The answer is: True" | video frames)`, gathered from `log_softmax(logits)` of a vision-language model. Higher = the VLM thinks the video completes the caption.
- **Progress curve** — same reward recomputed on trajectory prefixes (first k frames, k uniformly spaced), then min-max normalized to [0, 1].
- **VOC** — Spearman correlation of the progress curve vs. chronological order. A monotonically rising curve over a successful trajectory → VOC near 1.

## Files

- `topreward_test.py` — the experiment script. Self-contained loader for `mderry/ego4d-manipulation-v1` (LeRobot v3.0 format, no `lerobot` dep), Qwen-VL-style logits reward, progress curve, per-sample plots.
- `compare_models.py` — side-by-side comparison across multiple VLMs.
- `run_topreward_test.sbatch` — Slurm submission script (gitignored — cluster-specific).

## External paths

This dir is the working directory at runtime, but it depends on two sibling paths under the parent TOPReward checkout:

| Path | Purpose |
| --- | --- |
| `<TOPREWARD_ROOT>/repo/.venv_tr/` | Python venv built from the upstream TOPReward repo (torch, transformers, qwen-vl-utils, PyAV, pyarrow, …). Activated via `source`. Nothing is imported from the repo source — only its installed packages. |
| `<TOPREWARD_ROOT>/hf_cache/` | `HF_HOME` target. Model weights (Qwen3-VL ≈ 8 GB) and Ego4D LeRobot shards land here so they can be shared across runs and kept off `$HOME`. |

Layout this expects:

```
<TOPREWARD_ROOT>/
├── repo/                       # cloned topreward repo, with .venv_tr/ provisioned
├── hf_cache/                   # HF_HOME for weights + datasets
└── ego4d_toprewards_experiments/   # this directory
```

Both paths are referenced as absolute paths from the sbatch; the local recipe below sets them inline.

## Install

Managed with [uv](https://github.com/astral-sh/uv). One command to create `.venv/` and install everything from `uv.lock`:

```bash
uv sync
```

To run anything: prefix with `uv run` (it auto-activates `.venv/`):

```bash
uv run python topreward_test.py --help
uv run pyright topreward_test.py compare_models.py
```
## Run

Stand-alone (uses the experiments dir's own venv from `uv sync`):

```bash
cd <TOPREWARD_ROOT>/ego4d_toprewards_experiments
export HF_HOME=<TOPREWARD_ROOT>/hf_cache    # optional, see [External paths](#external-paths)
uv run python topreward_test.py --num-samples 40 \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --out runs/Qwen_Qwen3-VL-4B-Instruct/topreward.jsonl \
    --plots-dir runs/Qwen_Qwen3-VL-4B-Instruct/plots
```

Reusing the upstream's existing venv (skip `uv sync`):

```bash
cd <TOPREWARD_ROOT>/ego4d_toprewards_experiments
source <TOPREWARD_ROOT>/repo/.venv_tr/bin/activate
export HF_HOME=<TOPREWARD_ROOT>/hf_cache
python topreward_test.py --num-samples 40 --model Qwen/Qwen3-VL-4B-Instruct \
    --out runs/Qwen_Qwen3-VL-4B-Instruct/topreward.jsonl \
    --plots-dir runs/Qwen_Qwen3-VL-4B-Instruct/plots
```

Slurm:

```bash
NUM_SAMPLES=200 MODEL=Qwen/Qwen3-VL-4B-Instruct sbatch run_topreward_test.sbatch
```

Environment variables read by the sbatch: `MODEL`, `NUM_SAMPLES`, `MAX_FRAMES`, `NUM_PREFIXES`.

## Outputs

Per-model directory under `runs/<tag>/`:

- `topreward.jsonl` — one record per clip (reward, per-token log-probs, prefix rewards, progress curve, VOC).
- `topreward.summary.json` — aggregate stats (mean/std/min/max for reward, P(True), VOC).
- `plots/ep<NNNN>.png` — progress curve + keyframe strip per clip.

## Dataset

[`mderry/ego4d-manipulation-v1`](https://huggingface.co/datasets/mderry/ego4d-manipulation-v1) — Ego4D manipulation episodes re-packaged in LeRobot v3.0 format (egocentric mp4 + narration captions in `meta/tasks.parquet`). Avoids the gated Ego4D download.
