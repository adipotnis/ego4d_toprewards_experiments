"""topreward_test.py — single-file TOPReward-style reward, tested on Ego4D.

This is a self-contained re-implementation of the core TOPReward idea
("Token Probabilities as Zero-Shot Rewards"): use a vision-language model and
read the answer **logits** to score how well a video matches a text caption,
instead of asking the model to emit a number.

For every Ego4D clip (real first-person video + its narration caption) we
compute two things the user asked for:

  * LOGITS reward
      Build the prompt  [video frames] + "... completes the following task:
      {caption} ... The answer is: True", run a single forward pass, and read
      the model's output logits. The reward is the per-token log-probability
      (log_softmax of the logits, gathered at the gold token ids) of the
      answer span. Higher == the model is more confident the video completes
      the caption. This is exactly the signal TOPReward uses
      (see topreward/clients/qwen.py::compute_instruction_reward).

  * PROGRESS curve
      Recompute the logits reward on trajectory *prefixes* (the first k frames,
      for k uniformly spaced over the clip), then min-max normalise the reward
      curve to [0, 1]. A well-behaved reward rises as more of the task is shown,
      so this normalised curve is the model's predicted task-completion
      ("progress") over time. We summarise its monotonicity with VOC
      (Value-Order Correlation = Spearman corr. of the curve vs. chronological
      order), the same metric used in topreward/metrics/voc.py.

Data
----
mderry/ego4d-manipulation-v1 — Ego4D re-packaged in LeRobot v3.0 format. It
bundles the egocentric mp4 video together with a per-episode task narration,
so no gated Ego4D download is required. Each episode maps to a [start, end]
frame range inside a chunked mp4; captions live in meta/tasks.parquet.

Run
---
    python tests/topreward_test.py --num-samples 40 \
        --model Qwen/Qwen3-VL-4B-Instruct --out runs/ego4d_topreward.jsonl

It is named *topreward_test.py (not test_*.py) on purpose so pytest does not
collect it — it is a runnable experiment script, not a unit test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Constants for the Ego4D / LeRobot-v3.0 source dataset.
# ----------------------------------------------------------------------------
EGO4D_REPO = "mderry/ego4d-manipulation-v1"
EGO4D_SUBROOT = "0000"  # the dataset nests several LeRobot roots; use the first
EGO4D_FPS = 10.0  # native fps from 0000/meta/info.json
VIDEO_KEY = "observation.images.ego"

# The text scaffold around the caption. Mirrors topreward/clients/qwen.py.
PROMPT_PREFIX = "The above video shows a first-person trajectory that completes the following task: "
ANSWER_LEADIN = " Decide whether the above statement is True or not. The answer is:"
ANSWER_WORD = " True"


# ----------------------------------------------------------------------------
# Data structures.
# ----------------------------------------------------------------------------
@dataclass
class Ego4DSample:
    episode_index: int
    task_index: int
    caption: str
    frames: list  # list[np.ndarray HWC uint8], chronological order
    fps: float


@dataclass
class SampleResult:
    episode_index: int
    task_index: int
    caption: str
    num_frames: int
    # --- logits reward (full clip) ---
    reward_mean: float  # mean per-token log-prob of the answer span
    reward_sum: float  # summed log-prob
    answer_token_logprob: float  # log-prob of the decisive " True" token
    answer_token_prob: float  # exp() of the above, in [0, 1]
    token_count: int
    per_token_log_probs: list = field(default_factory=list)
    # --- progress curve ---
    prefix_frame_counts: list = field(default_factory=list)
    prefix_rewards: list = field(default_factory=list)  # raw mean log-prob per prefix
    progress: list = field(default_factory=list)  # min-max normalised -> [0,1]
    voc: float = float("nan")  # Spearman(progress, chronological order)
    error: str | None = None


# ----------------------------------------------------------------------------
# Ego4D loading (self-contained; no lerobot dependency).
# ----------------------------------------------------------------------------
def _read_parquet(path: str) -> dict:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pydict()


def _decode_frames(video_path: str, start: int, end: int) -> list:
    """Decode frames [start, end] (inclusive) from an mp4 using PyAV.

    Frame indices are relative to the given video file (verified against the
    dataset: each video file restarts its frame numbering at 0).
    """
    import av

    frames: list = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        idx = 0
        for frame in container.decode(stream):
            if idx > end:
                break
            if idx >= start:
                frames.append(frame.to_ndarray(format="rgb24"))
            idx += 1
    return frames


def load_ego4d_samples(num_samples: int, cache_dir: str) -> list[Ego4DSample]:
    """Download metadata + the needed video files and build samples.

    Walks episode-meta files in order, collecting episodes until we have
    `num_samples`, downloading each referenced video file on demand.
    """
    from huggingface_hub import hf_hub_download

    def dl(rel: str) -> str:
        return hf_hub_download(
            repo_id=EGO4D_REPO,
            filename=f"{EGO4D_SUBROOT}/{rel}",
            repo_type="dataset",
            cache_dir=cache_dir,
        )

    # task_index -> caption
    tasks = _read_parquet(dl("meta/tasks.parquet"))
    caption_of = dict(zip(tasks["task_index"], tasks["task"], strict=False))

    samples: list[Ego4DSample] = []
    video_cache: dict[int, str] = {}
    meta_file_idx = 0
    while len(samples) < num_samples:
        try:
            ep_path = dl(f"meta/episodes/chunk-000/file-{meta_file_idx:03d}.parquet")
        except Exception as exc:  # noqa: BLE001 - ran out of episode files
            print(f"[load] stopping at meta file {meta_file_idx}: {exc}")
            break
        ep = _read_parquet(ep_path)
        n = len(ep["episode_index"])
        for i in range(n):
            if len(samples) >= num_samples:
                break
            vfi = int(ep["video_file_index"][i])
            vci = int(ep["video_chunk_index"][i])
            if vfi not in video_cache:
                video_cache[vfi] = dl(f"videos/{VIDEO_KEY}/chunk-{vci:03d}/file-{vfi:03d}.mp4")
            start, end = int(ep["start_frame"][i]), int(ep["end_frame"][i])
            frames = _decode_frames(video_cache[vfi], start, end)
            if len(frames) < 2:
                continue  # need >=2 frames for a trajectory
            ti = int(ep["task_index"][i])
            caption = caption_of.get(ti, "").replace(" #Unsure", "").strip()
            if not caption:
                continue
            samples.append(
                Ego4DSample(
                    episode_index=int(ep["episode_index"][i]),
                    task_index=ti,
                    caption=caption,
                    frames=frames,
                    fps=EGO4D_FPS,
                )
            )
            print(f"[load] sample {len(samples):>2}/{num_samples} ep={ep['episode_index'][i]} frames={len(frames)} caption={caption[:60]!r}")
        meta_file_idx += 1
    return samples


def _uniform_subsample(frames: list, n: int) -> list:
    """Return n uniformly spaced frames (chronological order preserved)."""
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


# ----------------------------------------------------------------------------
# The TOPReward core: read the answer logits.
# ----------------------------------------------------------------------------
def _to_pil(frame):
    from PIL import Image

    return Image.fromarray(frame)


def _vision_info(messages):
    """qwen_vl_utils.process_vision_info, tolerant of 2- or 3-tuple returns."""
    from qwen_vl_utils import process_vision_info

    out = process_vision_info(messages)
    if isinstance(out, tuple) and len(out) == 3:
        image_inputs, video_inputs, video_kwargs = out
    else:
        image_inputs, video_inputs = out
        video_kwargs = {}
    return image_inputs, video_inputs, video_kwargs


def _is_molmo(model_name: str) -> bool:
    """Molmo2 needs a different vision pipeline (molmo_utils + timestamps)."""
    return "molmo" in model_name.lower()


def _strip_trailing_eos(processor, prompt_chat: str) -> str:
    """Strip only the FINAL turn-closing eos so the scored text continues the
    user turn. Using rsplit (last eos) — not split (first eos) — is crucial for
    templates that prepend a system message (e.g. Qwen2.5-VL); splitting on the
    first eos would discard the user turn and all vision tokens."""
    eos = getattr(processor.tokenizer, "eos_token", None)
    if eos and eos in prompt_chat:
        return prompt_chat.rsplit(eos, 1)[0]
    return prompt_chat


def _qwen_inputs(model, processor, pil_frames: list, fps: float, prompt_text: str, scored_text: str):
    """Build (forward inputs, prompt_len) for a Qwen-VL-style processor.

    Renders the chat prefix (video + prompt_text) WITHOUT a generation prompt,
    strips a trailing eos, then appends the span we want to score. Mirrors
    topreward/clients/qwen.py.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames, "fps": fps},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    prompt_chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_chat = _strip_trailing_eos(processor, prompt_chat)
    full_text = f"{prompt_chat}{scored_text}"

    image_inputs, video_inputs, video_kwargs = _vision_info(messages)
    inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)
    prompt_inputs = processor(
        text=[prompt_chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    return inputs, prompt_len


def _molmo_inputs(model, processor, pil_frames: list, prompt_text: str, scored_text: str):
    """Build (forward inputs, prompt_len) for a Molmo2 processor.

    Molmo2 uses molmo_utils.process_vision_info (timestamp-based video metadata)
    rather than qwen_vl_utils, and the processor takes videos= + video_metadata=
    with the text as a plain string. Mirrors topreward/clients/molmo.py. The
    answer span scored is the same as the Qwen path, so VOC/reward stay comparable.
    """
    try:
        from molmo_utils import process_vision_info as molmo_vision_info
    except ImportError as exc:
        raise ImportError(
            "molmo_utils is required to run Molmo2 models. Install it with "
            "`uv add molmo_utils` (or `pip install molmo-utils`)."
        ) from exc

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames, "timestamps": np.arange(len(pil_frames))},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    _, videos, video_kwargs = molmo_vision_info(messages)
    if videos is None:
        raise ValueError("molmo_utils.process_vision_info returned no videos")
    videos, video_metadatas = zip(*videos, strict=False)
    videos, video_metadatas = list(videos), list(video_metadatas)
    # Single-frame prefixes confuse the timestamp metadata; patch as molmo.py does.
    for idx, md in enumerate(video_metadatas):
        if md["total_num_frames"] == 1:
            video_metadatas[idx]["fps"] = 1.0
            video_metadatas[idx]["frames_indices"] = np.array([1])

    prompt_chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_chat = _strip_trailing_eos(processor, prompt_chat)
    full_text = f"{prompt_chat}{scored_text}"

    def _proc(text: str):
        out = processor(
            videos=videos,
            video_metadata=video_metadatas,
            text=text,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        return {k: v.to(model.device) for k, v in out.items()}

    inputs = _proc(full_text)
    prompt_inputs = _proc(prompt_chat)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    return inputs, prompt_len


def logits_reward(model, processor, frames: list, caption: str, fps: float, reduction: str = "mean", is_molmo: bool = False) -> dict:
    """Score a (video, caption) pair by the log-prob of the answer span.

    Builds  [video] + PROMPT_PREFIX + caption + ANSWER_LEADIN + " True", runs a
    single forward pass, and reads log_softmax(logits) at the gold tokens of the
    answer span (the caption restatement + the decisive " True"). This is the
    TOPReward "token probabilities as rewards" signal.
    """
    import torch
    import torch.nn.functional as F

    pil_frames = [_to_pil(f) for f in frames]
    prompt_text = PROMPT_PREFIX
    # Everything after the video that we want to *score* under the model logits.
    scored_text = f"{caption}{ANSWER_LEADIN}{ANSWER_WORD}"

    # Build the forward-pass inputs and the prompt-prefix length (how many leading
    # tokens to mask out — we only score the answer span, not the video/prompt).
    # The reward definition is identical across model families; only the vision /
    # processor plumbing differs (Qwen-VL vs Molmo2), so they share the masking and
    # log-prob math below.
    if is_molmo:
        inputs, prompt_len = _molmo_inputs(model, processor, pil_frames, prompt_text, scored_text)
    else:
        inputs, prompt_len = _qwen_inputs(model, processor, pil_frames, fps, prompt_text, scored_text)

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    if "attention_mask" in inputs:
        labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # log p(token_t | < t)  -> shift by one.
    logits = outputs.logits[:, :-1, :]
    target = labels[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    mask = target != -100
    safe = target.masked_fill(~mask, 0)
    tok_lp = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    masked = tok_lp[mask]

    if masked.numel() == 0:
        raise ValueError("empty answer span after masking")

    per_token = masked.detach().cpu().tolist()
    reward_mean = float(masked.mean().item())
    reward_sum = float(masked.sum().item())
    answer_lp = per_token[-1]  # the decisive " True" token is scored last
    reward = reward_sum if reduction == "sum" else reward_mean
    return {
        "reward": reward,
        "reward_mean": reward_mean,
        "reward_sum": reward_sum,
        "answer_token_logprob": float(answer_lp),
        "answer_token_prob": float(np.exp(answer_lp)),
        "token_count": len(per_token),
        "per_token_log_probs": per_token,
    }


def progress_curve(model, processor, frames: list, caption: str, fps: float, num_prefixes: int = 8, is_molmo: bool = False) -> dict:
    """Compute the reward on trajectory prefixes and normalise -> progress.

    Mirrors topreward/clients/qwen.py::compute_instruction_rewards_for_prefixes:
    score the first k frames for k uniformly spaced in [2, N], then min-max
    normalise the reward curve to [0, 1].
    """
    from scipy.stats import spearmanr

    n = len(frames)
    lengths = sorted({int(x) for x in np.linspace(2, n, min(num_prefixes, n - 1))})
    rewards = []
    for k in lengths:
        r = logits_reward(model, processor, frames[:k], caption, fps, reduction="mean", is_molmo=is_molmo)
        rewards.append(r["reward_mean"])

    arr = np.asarray(rewards, dtype=float)
    if arr.size >= 2 and not np.allclose(arr, arr[0]):
        progress = ((arr - arr.min()) / (arr.max() - arr.min())).tolist()
    else:
        progress = np.ones_like(arr).tolist()

    if len(progress) >= 2 and not np.allclose(progress, progress[0]):
        voc = float(spearmanr(progress, np.arange(len(progress))).statistic)
    else:
        voc = float("nan")

    return {"prefix_frame_counts": lengths, "prefix_rewards": rewards, "progress": progress, "voc": voc}


# ----------------------------------------------------------------------------
# Plotting: progress curve with the corresponding keyframes alongside.
# ----------------------------------------------------------------------------
def _wrap(text: str, width: int = 70) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width)[:3])


def plot_sample(res: "SampleResult", frames: list, out_png: Path, max_keyframes: int = 6) -> None:
    """Save a figure: the progress curve on the left, the keyframes that each
    prefix ends on shown as a vertical strip on the right side.

    `frames` is the (subsampled) chronological frame list actually fed to the
    model; prefix k ends on frames[k-1].
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = res.prefix_frame_counts
    progress = res.progress
    if not counts or not progress:
        return

    # Choose up to `max_keyframes` prefix points (evenly) to display as images.
    sel = list(range(len(counts)))
    if len(sel) > max_keyframes:
        sel = [int(round(x)) for x in np.linspace(0, len(counts) - 1, max_keyframes)]
        sel = sorted(set(sel))
    n_kf = len(sel)

    fig = plt.figure(figsize=(13, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(n_kf, 3)

    # --- progress curve (left 2/3) ---
    ax = fig.add_subplot(gs[:, :2])
    ax.plot(counts, progress, "-o", color="#1f77b4", lw=2, ms=6, label="progress (norm. logit reward)")
    ax.set_xlabel("prefix length (# frames shown)")
    ax.set_ylabel("predicted progress  [0, 1]")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"ep {res.episode_index} | VOC={res.voc:.3f} | "
        f"reward_mean={res.reward_mean:.3f} | P(True)={res.answer_token_prob:.3f}\n{_wrap(res.caption)}",
        fontsize=9,
        loc="left",
    )
    # mark the keyframe x-positions
    for j in sel:
        ax.axvline(counts[j], color="grey", ls=":", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)

    # --- keyframe strip (right 1/3) ---
    for row, j in enumerate(sel):
        axk = fig.add_subplot(gs[row, 2])
        fidx = min(counts[j] - 1, len(frames) - 1)
        axk.imshow(frames[fidx])
        axk.set_xticks([])
        axk.set_yticks([])
        axk.set_ylabel(f"{counts[j]}f\n{progress[j]:.2f}", fontsize=7, rotation=0, labelpad=16, va="center")

    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Model loading + orchestration.
# ----------------------------------------------------------------------------
def load_model(model_name: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[model] loading {model_name} ...")
    common = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
    if _is_molmo(model_name):
        # Molmo2 ships custom modeling code (trust_remote_code) and prefers
        # flash-attention-2; degrade through sdpa to eager so the run still works
        # wherever flash-attn isn't built or the custom model rejects a kernel.
        model = None
        for attn in ("flash_attention_2", "sdpa", "eager"):
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name, trust_remote_code=True, attn_implementation=attn, **common
                )
                print(f"[model] loaded with attn_implementation={attn}")
                break
            except Exception as exc:  # noqa: BLE001 - try the next available kernel
                print(f"[model] attn_implementation={attn} unavailable ({exc!r})")
        if model is None:
            raise RuntimeError(f"could not load {model_name} with any attention implementation")
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_name, attn_implementation="sdpa", **common)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print(f"[model] loaded on {model.device}")
    return model, processor


def run(num_samples: int, model_name: str, out_path: str, cache_dir: str, max_frames: int, num_prefixes: int, plots_dir: str | None) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plots = Path(plots_dir) if plots_dir else None
    if plots:
        plots.mkdir(parents=True, exist_ok=True)

    samples = load_ego4d_samples(num_samples, cache_dir=cache_dir)
    print(f"[run] loaded {len(samples)} Ego4D samples")

    model, processor = load_model(model_name)
    is_molmo = _is_molmo(model_name)

    results: list[SampleResult] = []
    with out.open("w", encoding="utf-8") as fh:
        for i, s in enumerate(samples):
            frames = _uniform_subsample(s.frames, max_frames)
            print(f"[run] {i + 1}/{len(samples)} ep={s.episode_index} frames={len(frames)} :: {s.caption[:70]!r}")
            try:
                rw = logits_reward(model, processor, frames, s.caption, s.fps, reduction="mean", is_molmo=is_molmo)
                pg = progress_curve(model, processor, frames, s.caption, s.fps, num_prefixes=num_prefixes, is_molmo=is_molmo)
                res = SampleResult(
                    episode_index=s.episode_index,
                    task_index=s.task_index,
                    caption=s.caption,
                    num_frames=len(frames),
                    reward_mean=rw["reward_mean"],
                    reward_sum=rw["reward_sum"],
                    answer_token_logprob=rw["answer_token_logprob"],
                    answer_token_prob=rw["answer_token_prob"],
                    token_count=rw["token_count"],
                    per_token_log_probs=rw["per_token_log_probs"],
                    prefix_frame_counts=pg["prefix_frame_counts"],
                    prefix_rewards=pg["prefix_rewards"],
                    progress=pg["progress"],
                    voc=pg["voc"],
                )
                print(f"        reward_mean={res.reward_mean:.4f}  P(True)={res.answer_token_prob:.4f}  VOC={res.voc:.4f}")
                if plots is not None:
                    try:
                        plot_sample(res, frames, plots / f"ep{res.episode_index:04d}.png")
                    except Exception as pexc:  # noqa: BLE001 - plotting must not kill the run
                        print(f"        plot failed: {pexc!r}")
            except Exception as exc:  # noqa: BLE001 - record and continue
                res = SampleResult(
                    episode_index=s.episode_index,
                    task_index=s.task_index,
                    caption=s.caption,
                    num_frames=len(frames),
                    reward_mean=float("nan"),
                    reward_sum=float("nan"),
                    answer_token_logprob=float("nan"),
                    answer_token_prob=float("nan"),
                    token_count=0,
                    error=repr(exc),
                )
                print(f"        ERROR: {exc!r}")
            results.append(res)
            fh.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
            fh.flush()

    _summarize(results, model_name, out)


def _summarize(results: list[SampleResult], model_name: str, out: Path) -> None:
    ok = [r for r in results if r.error is None]
    rewards = np.array([r.reward_mean for r in ok], dtype=float)
    probs = np.array([r.answer_token_prob for r in ok], dtype=float)
    vocs = np.array([r.voc for r in ok if not np.isnan(r.voc)], dtype=float)

    def stats(a):
        a = a[~np.isnan(a)]
        if a.size == 0:
            return {"n": 0}
        return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}

    summary = {
        "model": model_name,
        "dataset": EGO4D_REPO,
        "num_samples": len(results),
        "num_valid": len(ok),
        "num_errors": len(results) - len(ok),
        "timestamp": datetime.now().isoformat(),
        "reward_mean_stats": stats(rewards),
        "answer_prob_stats": stats(probs),
        "voc_stats": stats(vocs),
    }
    sp = out.with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, indent=2))
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\n[done] per-sample -> {out}\n[done] summary    -> {sp}")


def main() -> None:
    p = argparse.ArgumentParser(description="TOPReward-style logits+progress reward on Ego4D")
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--out", default="runs/ego4d_topreward.jsonl")
    p.add_argument("--cache-dir", default=None, help="HF datasets cache dir (defaults to HF_HOME)")
    p.add_argument("--max-frames", type=int, default=12, help="max frames per clip fed to the model")
    p.add_argument("--num-prefixes", type=int, default=8, help="prefix points for the progress curve")
    p.add_argument("--plots-dir", default="runs/plots", help="dir for per-sample progress+keyframe plots ('' to disable)")
    args = p.parse_args()
    run(
        num_samples=args.num_samples,
        model_name=args.model,
        out_path=args.out,
        cache_dir=args.cache_dir,
        max_frames=args.max_frames,
        num_prefixes=args.num_prefixes,
        plots_dir=args.plots_dir or None,
    )


if __name__ == "__main__":
    main()
