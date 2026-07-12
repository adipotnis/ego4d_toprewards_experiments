# Ego4D TOPReward Experiments

Standalone re-implementation of [TOPReward](https://topreward.github.io/webpage/) ("Token Probabilities as Hidden Zero-Shot Rewards") run on Ego4D first-person manipulation clips.

Self-contained: does not import from the upstream `topreward` Python package. Has its own `pyproject.toml` / `uv.lock`, so the experiments dir can stand on its own with `uv sync`. (The cluster sbatch happens to reuse the upstream's `.venv_tr/` because it's already provisioned — see [External paths](#external-paths) — but that's an optimization, not a requirement.)

## What it does

For every Ego4D clip (egocentric video + narration caption) the script computes:

- **Logits reward** — `log P(caption + " The answer is: True" | video frames)`, gathered from `log_softmax(logits)` of a vision-language model. Higher = the VLM thinks the video completes the caption.
- **Progress curve** — same reward recomputed on trajectory prefixes (first k frames, k uniformly spaced), then min-max normalized to [0, 1].
- **VOC** — Spearman correlation of the progress curve vs. chronological order. A monotonically rising curve over a successful trajectory → VOC near 1.

## Files

- `topreward_test.py` — the experiment script. Self-contained loader for `mderry/ego4d-manipulation-v1` (LeRobot v3.0 format, no `lerobot` dep), logits reward, progress curve, per-sample plots. Supports both **Qwen3-VL** (via `qwen-vl-utils`) and **Molmo2** (via `molmo-utils`, timestamp-based video metadata) — the model family is auto-detected from the `--model` id (anything containing `molmo` takes the Molmo2 path). Both families score the same answer span, so VOC / reward stay directly comparable.
- `compare_models.py` — side-by-side comparison across multiple VLMs.
- `make_subgoal_videos.py` — renders a synced mp4 per episode for a `--split-subgoals` run: the real clip plays on top while a cursor sweeps each subgoal's progress curve (shown side by side below) in sync. Encodes via PyAV, no ffmpeg binary needed.
- `run_topreward_test.sbatch` — Slurm submission script (gitignored — cluster-specific).

### Per-subgoal mode

Ego4D captions are sequences of atomic sub-actions ("subgoals") joined by a comma
(most end each with a period, `"., "`; some use a plain `", "`). Passing
`--split-subgoals` (env `SPLIT_SUBGOALS=1` for the sbatch) splits each caption into
its subgoals and scores/plots **each subgoal independently over the whole clip** —
there are no per-subgoal frame boundaries in the data, so every subgoal gets its own
progress curve + VOC. Output goes to a `..._subgoals/` dir: per-episode JSONL
(`EpisodeResult` with a list of `SubgoalResult`), per-subgoal plots
(`epNNNN_sgMM.png`), and one overlay per episode (`epNNNN_overlay.png`). The original
per-episode path is unchanged.

```bash
NUM_SAMPLES=12 MODEL=Qwen/Qwen3-VL-8B-Instruct SPLIT_SUBGOALS=1 sbatch run_topreward_test.sbatch
# then, on any node (CPU only):
python make_subgoal_videos.py --jsonl runs/Qwen_Qwen3-VL-8B-Instruct_subgoals/topreward.jsonl
```

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

### Molmo2

Pass a Molmo2 model id and the script auto-selects the Molmo2 vision path (no code/flag change):

```bash
uv run python topreward_test.py --num-samples 40 \
    --model allenai/Molmo2-8B \
    --out runs/allenai_Molmo2-8B/topreward.jsonl \
    --plots-dir runs/allenai_Molmo2-8B/plots
# or on Slurm:
NUM_SAMPLES=200 MODEL=allenai/Molmo2-8B sbatch run_topreward_test.sbatch
```

Molmo2 needs the `molmo-utils` package (a declared dependency, installed by `uv sync`) and ships custom modeling code, so it loads with `trust_remote_code=True`. It prefers flash-attention-2 and falls back to `sdpa` when flash-attn isn't built. If you reuse the upstream `.venv_tr/` rather than `uv sync`, make sure `molmo_utils` is installed there too (`allenai/Molmo2-4B` is the smaller, faster checkpoint).

## Outputs

Per-model directory under `runs/<tag>/`:

- `topreward.jsonl` — one record per clip (reward, per-token log-probs, prefix rewards, progress curve, VOC).
- `topreward.summary.json` — aggregate stats (mean/std/min/max for reward, P(True), VOC).
- `plots/ep<NNNN>.png` — progress curve + keyframe strip per clip.

## Dataset

[`mderry/ego4d-manipulation-v1`](https://huggingface.co/datasets/mderry/ego4d-manipulation-v1) — Ego4D manipulation episodes re-packaged in LeRobot v3.0 format (egocentric mp4 + narration captions in `meta/tasks.parquet`). Avoids the gated Ego4D download.
