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
    python topreward_test.py --num-samples 40 \
        --model Qwen/Qwen3-VL-4B-Instruct --out runs/ego4d_topreward.jsonl

It is named *topreward_test.py (not test_*.py) on purpose so pytest does not
collect it — it is a runnable experiment script, not a unit test.

`make_subgoal_videos.py` and `compare_models.py` import the small shared helpers
defined here (loader, subsampling, JSONL/stats/plot utilities); the heavy
imports (torch, transformers, av, matplotlib) are all lazy so importing this
module is cheap.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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

NAN = float("nan")


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


# The reward fields below default to NaN/empty so a failed clip is recorded as
# `Result(<identity>, error=repr(exc))`. Their names match the keys returned by
# `score_clip`, so a successful clip is `Result(<identity>, **score_clip(...))`.
@dataclass
class SubgoalResult:
    """One sub-action of an episode caption, scored on its own over the full clip.

    Field-compatible with `plot_sample` (episode_index, caption, voc, reward_mean,
    answer_token_prob, prefix_frame_counts, progress) so the existing per-sample
    plotter renders a subgoal without changes. `caption` holds the subgoal text.
    """

    episode_index: int
    subgoal_index: int
    caption: str  # the subgoal (sub-action) text
    num_frames: int
    reward_mean: float = NAN
    reward_sum: float = NAN
    answer_token_logprob: float = NAN
    answer_token_prob: float = NAN
    token_count: int = 0
    per_token_log_probs: list = field(default_factory=list)
    prefix_frame_counts: list = field(default_factory=list)
    prefix_rewards: list = field(default_factory=list)
    progress: list = field(default_factory=list)
    voc: float = NAN
    error: str | None = None


@dataclass
class EpisodeResult:
    """An episode's full caption split into subgoals, each scored independently."""

    episode_index: int
    task_index: int
    full_caption: str
    num_frames: int
    num_subgoals: int
    subgoals: list = field(default_factory=list)  # list[SubgoalResult]


@dataclass
class SampleResult:
    episode_index: int
    task_index: int
    caption: str
    num_frames: int
    # --- logits reward (full clip) ---
    reward_mean: float = NAN  # mean per-token log-prob of the answer span
    reward_sum: float = NAN  # summed log-prob
    answer_token_logprob: float = NAN  # log-prob of the decisive " True" token
    answer_token_prob: float = NAN  # exp() of the above, in [0, 1]
    token_count: int = 0
    per_token_log_probs: list = field(default_factory=list)
    # --- progress curve ---
    prefix_frame_counts: list = field(default_factory=list)
    prefix_rewards: list = field(default_factory=list)  # raw mean log-prob per prefix
    progress: list = field(default_factory=list)  # min-max normalised -> [0,1]
    voc: float = NAN  # Spearman(progress, chronological order)
    error: str | None = None


# ----------------------------------------------------------------------------
# Small shared utilities (also used by make_subgoal_videos.py / compare_models.py).
# ----------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summary_stats(vals: Iterable[float | None]) -> dict:
    """n / mean / std / min / max over the finite values (None and NaN dropped)."""
    a = np.asarray(list(vals), dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"n": 0, "mean": NAN, "std": NAN, "min": NAN, "max": NAN}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def uniform_subsample(seq: list, n: int) -> list:
    """Return n uniformly spaced elements (order preserved); all of them if len <= n."""
    if len(seq) <= n:
        return seq
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return [seq[i] for i in idx]


def wrap_title(text: str, width: int = 70, max_lines: int = 3) -> str:
    return "\n".join(textwrap.wrap(text, width=width, max_lines=max_lines, placeholder=" …"))


def agg_pyplot():
    """matplotlib.pyplot with the headless Agg backend selected."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def style_progress_axis(ax) -> None:
    ax.set_xlabel("prefix length (# frames shown)")
    ax.set_ylabel("predicted progress  [0, 1]")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------------------
# Ego4D loading (self-contained; no lerobot dependency).
# ----------------------------------------------------------------------------
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
        for idx, frame in enumerate(container.decode(stream)):
            if idx > end:
                break
            if idx >= start:
                frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def _iter_episode_rows(dl: Callable[[str], str]) -> Iterator[dict]:
    """Yield episode-meta rows in dataset order, walking the meta parquet files."""
    import pyarrow.parquet as pq

    meta_file_idx = 0
    while True:
        try:
            ep_path = dl(f"meta/episodes/chunk-000/file-{meta_file_idx:03d}.parquet")
        except Exception as exc:  # ran out of episode files
            print(f"[load] stopping at meta file {meta_file_idx}: {exc}")
            return
        yield from pq.read_table(ep_path).to_pylist()
        meta_file_idx += 1


def load_ego4d_samples(
    num_samples: int,
    cache_dir: str | None,
    *,
    max_frames: int | None = None,
    episode_indices: set[int] | None = None,
) -> list[Ego4DSample]:
    """Download metadata + the needed video files and build samples.

    Walks episodes in dataset order, decoding each referenced video file on
    demand, until `num_samples` usable episodes (>= 2 frames, non-empty caption)
    are collected. With `episode_indices`, only those episodes are decoded and
    the walk stops once all of them are found. With `max_frames`, each sample
    keeps only that many uniformly spaced frames.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def dl(rel: str) -> str:
        return hf_hub_download(
            repo_id=EGO4D_REPO,
            filename=f"{EGO4D_SUBROOT}/{rel}",
            repo_type="dataset",
            cache_dir=cache_dir,
        )

    # task_index -> caption
    tasks = pq.read_table(dl("meta/tasks.parquet")).to_pydict()
    caption_of = dict(zip(tasks["task_index"], tasks["task"], strict=False))

    samples: list[Ego4DSample] = []
    video_cache: dict[int, str] = {}
    pending = set(episode_indices) if episode_indices is not None else None
    for row in _iter_episode_rows(dl):
        if len(samples) >= num_samples or (pending is not None and not pending):
            break
        epi = int(row["episode_index"])
        if pending is not None:
            if epi not in pending:
                continue
            pending.discard(epi)
        vfi, vci = int(row["video_file_index"]), int(row["video_chunk_index"])
        if vfi not in video_cache:
            video_cache[vfi] = dl(f"videos/{VIDEO_KEY}/chunk-{vci:03d}/file-{vfi:03d}.mp4")
        frames = _decode_frames(video_cache[vfi], int(row["start_frame"]), int(row["end_frame"]))
        if len(frames) < 2:
            continue  # need >=2 frames for a trajectory
        if max_frames is not None:
            frames = uniform_subsample(frames, max_frames)
        ti = int(row["task_index"])
        caption = caption_of.get(ti, "").replace(" #Unsure", "").strip()
        if not caption:
            continue
        samples.append(Ego4DSample(episode_index=epi, task_index=ti, caption=caption, frames=frames, fps=EGO4D_FPS))
        print(f"[load] sample {len(samples):>2}/{num_samples} ep={epi} frames={len(frames)} caption={caption[:60]!r}")
    return samples


def split_subgoals(caption: str) -> list[str]:
    """Split an Ego4D narration into its constituent sub-actions ("subgoals").

    Ego4D captions concatenate sub-actions with a comma; most end each
    sub-action with a period ("., "), but a chunk of the data uses a plain
    comma (", ") instead, e.g.
        "carries the bowl., touches the contents., pours the contents."
        "places the wand on the floor, picks the bottle, shakes the bottle"
    Both are handled by splitting on a comma with an optional preceding period.
    Each piece is normalised to end in a single ".".

        -> ["carries the bowl.", "touches the contents.", "pours the contents."]

    A caption with no comma yields one subgoal (the whole caption). Empty pieces
    are dropped; duplicate sub-actions are kept (they recur in the data and map
    to distinct moments in the video). Individual Ego4D sub-actions do not
    contain commas, so this does not over-split.
    """
    import re

    subgoals: list[str] = []
    for part in re.split(r"\.?\s*,\s*", caption):
        part = part.strip().rstrip(".").strip()
        if part:
            subgoals.append(part + ".")
    return subgoals


# ----------------------------------------------------------------------------
# Model families: everything that differs between Qwen-VL and Molmo2 lives here.
# The reward math below is family-agnostic and only sees a `VLM`.
# ----------------------------------------------------------------------------
@dataclass
class VLM:
    """A loaded model + processor and the family-specific forward-input builder."""

    model: Any
    processor: Any
    build_inputs: Callable[..., tuple[dict, int]]


@dataclass(frozen=True)
class _Family:
    match: str | None  # case-insensitive substring of the HF model id; None = default
    trust_remote_code: bool
    attn_candidates: tuple[str, ...]  # tried in order; first one that loads wins
    build_inputs: Callable[..., tuple[dict, int]]


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


def _strip_trailing_eos(processor, prompt_chat: str) -> str:
    """Strip only the FINAL turn-closing eos so the scored text continues the
    user turn. Using rsplit (last eos) — not split (first eos) — is crucial for
    templates that prepend a system message (e.g. Qwen2.5-VL); splitting on the
    first eos would discard the user turn and all vision tokens."""
    eos = getattr(processor.tokenizer, "eos_token", None)
    if eos and eos in prompt_chat:
        return prompt_chat.rsplit(eos, 1)[0]
    return prompt_chat


def _qwen_inputs(vlm: VLM, pil_frames: list, fps: float, prompt_text: str, scored_text: str):
    """Build (forward inputs, prompt_len) for a Qwen-VL-style processor.

    Renders the chat prefix (video + prompt_text) WITHOUT a generation prompt,
    strips a trailing eos, then appends the span we want to score. Mirrors
    topreward/clients/qwen.py.
    """
    processor = vlm.processor
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

    def _proc(text: str):
        return processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", **video_kwargs)

    inputs = _proc(full_text).to(vlm.model.device)
    prompt_len = int(_proc(prompt_chat)["input_ids"].shape[1])
    return inputs, prompt_len


def _molmo_inputs(vlm: VLM, pil_frames: list, fps: float, prompt_text: str, scored_text: str):
    """Build (forward inputs, prompt_len) for a Molmo2 processor.

    Molmo2 uses molmo_utils.process_vision_info (timestamp-based video metadata)
    rather than qwen_vl_utils, and the processor takes videos= + video_metadata=
    with the text as a plain string. Mirrors topreward/clients/molmo.py. The
    answer span scored is the same as the Qwen path, so VOC/reward stay comparable.
    `fps` is unused: Molmo2 takes frame-index timestamps instead.
    """
    try:
        from molmo_utils import process_vision_info as molmo_vision_info
    except ImportError as exc:
        raise ImportError("molmo_utils is required to run Molmo2 models. Install it with " "`uv add molmo_utils` (or `pip install molmo-utils`).") from exc

    processor = vlm.processor
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
    videos, video_metadatas = map(list, zip(*videos, strict=False))
    # Single-frame prefixes confuse the timestamp metadata; patch as molmo.py does.
    for md in video_metadatas:
        if md["total_num_frames"] == 1:
            md["fps"] = 1.0
            md["frames_indices"] = np.array([1])

    prompt_chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_chat = _strip_trailing_eos(processor, prompt_chat)
    full_text = f"{prompt_chat}{scored_text}"

    def _proc(text: str):
        return processor(videos=videos, video_metadata=video_metadatas, text=text, padding=True, return_tensors="pt", **video_kwargs)

    inputs = {k: v.to(vlm.model.device) for k, v in _proc(full_text).items()}
    prompt_len = int(_proc(prompt_chat)["input_ids"].shape[1])
    return inputs, prompt_len


# First matching family wins; the last entry (match=None) is the default.
_FAMILIES = (
    # Molmo2 ships custom modeling code and prefers flash-attention-2; degrade
    # through sdpa to eager so the run still works wherever flash-attn isn't
    # built or the custom model rejects a kernel.
    _Family(match="molmo", trust_remote_code=True, attn_candidates=("flash_attention_2", "sdpa", "eager"), build_inputs=_molmo_inputs),
    _Family(match=None, trust_remote_code=False, attn_candidates=("sdpa",), build_inputs=_qwen_inputs),
)


def _family_of(model_name: str) -> _Family:
    return next(f for f in _FAMILIES if f.match is None or f.match in model_name.lower())


def load_model(model_name: str) -> VLM:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    fam = _family_of(model_name)
    print(f"[model] loading {model_name} ...")
    last_exc: Exception | None = None
    for attn in fam.attn_candidates:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=fam.trust_remote_code,
                attn_implementation=attn,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
            print(f"[model] loaded with attn_implementation={attn}")
            break
        except Exception as exc:  # try the next available kernel
            print(f"[model] attn_implementation={attn} unavailable ({exc!r})")
            last_exc = exc
    else:
        raise RuntimeError(f"could not load {model_name} with any attention implementation") from last_exc
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print(f"[model] loaded on {model.device}")
    return VLM(model=model, processor=processor, build_inputs=fam.build_inputs)


# ----------------------------------------------------------------------------
# The TOPReward core: read the answer logits.
# ----------------------------------------------------------------------------
def logits_reward(vlm: VLM, frames: list, caption: str, fps: float) -> dict:
    """Score a (video, caption) pair by the log-prob of the answer span.

    Builds  [video] + PROMPT_PREFIX + caption + ANSWER_LEADIN + " True", runs a
    single forward pass, and reads log_softmax(logits) at the gold tokens of the
    answer span (the caption restatement + the decisive " True"). This is the
    TOPReward "token probabilities as rewards" signal.
    """
    import torch
    import torch.nn.functional as F

    pil_frames = [_to_pil(f) for f in frames]
    # Everything after the video that we want to *score* under the model logits.
    scored_text = f"{caption}{ANSWER_LEADIN}{ANSWER_WORD}"

    # Forward-pass inputs and the prompt-prefix length (how many leading tokens
    # to mask out — we only score the answer span, not the video/prompt).
    inputs, prompt_len = vlm.build_inputs(vlm, pil_frames, fps, PROMPT_PREFIX, scored_text)

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    if "attention_mask" in inputs:
        labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

    with torch.no_grad():
        outputs = vlm.model(**inputs)

    # log p(token_t | < t): the logits at position t-1 predict token t. Only the
    # answer span (t >= prompt_len) is scored, so slice it out *before* the
    # vocab-wide float32 log_softmax rather than materialising it for the whole
    # (video-token dominated) sequence.
    logits = outputs.logits[:, prompt_len - 1 : -1, :]
    target = labels[:, prompt_len:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    mask = target != -100
    safe = target.masked_fill(~mask, 0)
    tok_lp = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    masked = tok_lp[mask]

    if masked.numel() == 0:
        raise ValueError("empty answer span after masking")

    per_token = masked.detach().cpu().tolist()
    answer_lp = per_token[-1]  # the decisive " True" token is scored last
    return {
        "reward_mean": float(masked.mean().item()),
        "reward_sum": float(masked.sum().item()),
        "answer_token_logprob": float(answer_lp),
        "answer_token_prob": float(np.exp(answer_lp)),
        "token_count": len(per_token),
        "per_token_log_probs": per_token,
    }


def score_clip(vlm: VLM, frames: list, caption: str, fps: float, num_prefixes: int = 8) -> dict:
    """Full-clip logits reward + prefix progress curve for one (video, caption).

    Mirrors topreward/clients/qwen.py::compute_instruction_rewards_for_prefixes:
    score the first k frames for k uniformly spaced in [2, N], then min-max
    normalise the reward curve to [0, 1]. The last prefix is the full clip, so
    its pass doubles as the full-clip reward. Returns the union of the
    `logits_reward` keys and prefix_frame_counts / prefix_rewards / progress / voc.
    """
    from scipy.stats import spearmanr

    n = len(frames)
    lengths = sorted({int(x) for x in np.linspace(2, n, min(num_prefixes, n - 1))})
    per_prefix = [logits_reward(vlm, frames[:k], caption, fps) for k in lengths]
    rewards = [r["reward_mean"] for r in per_prefix]

    arr = np.asarray(rewards, dtype=float)
    if arr.size >= 2 and not np.allclose(arr, arr[0]):
        progress = ((arr - arr.min()) / (arr.max() - arr.min())).tolist()
        voc = float(spearmanr(progress, np.arange(arr.size)).statistic)
    else:
        progress, voc = np.ones_like(arr).tolist(), NAN

    return {**per_prefix[-1], "prefix_frame_counts": lengths, "prefix_rewards": rewards, "progress": progress, "voc": voc}


# ----------------------------------------------------------------------------
# Plotting: progress curve with the corresponding keyframes alongside.
# ----------------------------------------------------------------------------
def plot_sample(res: SampleResult | SubgoalResult, frames: list, out_png: Path, max_keyframes: int = 6) -> None:
    """Save a figure: the progress curve on the left, the keyframes that each
    prefix ends on shown as a vertical strip on the right side.

    `frames` is the (subsampled) chronological frame list actually fed to the
    model; prefix k ends on frames[k-1].
    """
    plt = agg_pyplot()

    counts = res.prefix_frame_counts
    progress = res.progress
    if not counts or not progress:
        return

    # Choose up to `max_keyframes` prefix points (evenly) to display as images.
    sel = uniform_subsample(list(range(len(counts))), max_keyframes)
    n_kf = len(sel)

    fig = plt.figure(figsize=(13, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(n_kf, 3)

    # --- progress curve (left 2/3) ---
    ax = fig.add_subplot(gs[:, :2])
    ax.plot(counts, progress, "-o", color="#1f77b4", lw=2, ms=6, label="progress (norm. logit reward)")
    style_progress_axis(ax)
    ax.set_title(
        f"ep {res.episode_index} | VOC={res.voc:.3f} | " f"reward_mean={res.reward_mean:.3f} | P(True)={res.answer_token_prob:.3f}\n{wrap_title(res.caption)}",
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


def plot_episode_overlay(ep: EpisodeResult, out_png: Path) -> None:
    """Overlay every subgoal's progress curve for one episode on shared axes.

    Each subgoal is scored independently over the same full clip, so the curves
    are directly comparable: which sub-action the model reads as "completing" as
    the video plays out, and how monotonic (VOC) each one is.
    """
    plt = agg_pyplot()

    subs = [s for s in ep.subgoals if s.error is None and s.prefix_frame_counts and s.progress]
    if not subs:
        return

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(subs):
        label = f"[{s.subgoal_index}] VOC={s.voc:.2f}  {s.caption[:48]}"
        ax.plot(s.prefix_frame_counts, s.progress, "-o", color=cmap(i % 10), lw=1.8, ms=4, label=label)
    style_progress_axis(ax)
    ax.set_title(
        f"ep {ep.episode_index} — {ep.num_subgoals} subgoals scored over the full clip",
        fontsize=10,
        loc="left",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, title="subgoal")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Orchestration.
# ----------------------------------------------------------------------------
def _prepare(num_samples: int, model_name: str, out_path: str, cache_dir: str | None, max_frames: int, plots_dir: str | None):
    """Shared run setup: output dirs, samples (already subsampled to max_frames), model."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plots = Path(plots_dir) if plots_dir else None
    if plots:
        plots.mkdir(parents=True, exist_ok=True)

    samples = load_ego4d_samples(num_samples, cache_dir=cache_dir, max_frames=max_frames)
    print(f"[run] loaded {len(samples)} Ego4D samples")

    vlm = load_model(model_name)
    return out, plots, samples, vlm


def _score_fields(vlm: VLM, frames: list, caption: str, fps: float, num_prefixes: int) -> dict:
    """`score_clip` that never raises: a failure becomes {"error": repr(exc)}."""
    try:
        fields = score_clip(vlm, frames, caption, fps, num_prefixes=num_prefixes)
    except Exception as exc:  # record and continue
        print(f"        ERROR: {exc!r}")
        return {"error": repr(exc)}
    print(f"        reward_mean={fields['reward_mean']:.4f}  P(True)={fields['answer_token_prob']:.4f}  VOC={fields['voc']:.4f}")
    return fields


def _try_plot(fn: Callable[..., None], *args, what: str = "plot") -> None:
    try:
        fn(*args)
    except Exception as exc:  # plotting must not kill the run
        print(f"        {what} failed: {exc!r}")


def _write_record(fh, record) -> None:
    fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    fh.flush()


def run(num_samples: int, model_name: str, out_path: str, cache_dir: str | None, max_frames: int, num_prefixes: int, plots_dir: str | None) -> None:
    out, plots, samples, vlm = _prepare(num_samples, model_name, out_path, cache_dir, max_frames, plots_dir)

    results: list[SampleResult] = []
    with out.open("w", encoding="utf-8") as fh:
        for i, s in enumerate(samples):
            print(f"[run] {i + 1}/{len(samples)} ep={s.episode_index} frames={len(s.frames)} :: {s.caption[:70]!r}")
            fields = _score_fields(vlm, s.frames, s.caption, s.fps, num_prefixes)
            res = SampleResult(episode_index=s.episode_index, task_index=s.task_index, caption=s.caption, num_frames=len(s.frames), **fields)
            if plots is not None and res.error is None:
                _try_plot(plot_sample, res, s.frames, plots / f"ep{res.episode_index:04d}.png")
            results.append(res)
            _write_record(fh, res)

    _summarize(results, model_name, out)


def run_subgoals(num_samples: int, model_name: str, out_path: str, cache_dir: str | None, max_frames: int, num_prefixes: int, plots_dir: str | None) -> None:
    """Per-subgoal variant of `run`: split each episode caption into sub-actions
    and score/plot each subgoal independently over the full clip.

    Writes one JSONL line per episode (an `EpisodeResult` with its subgoals), and
    for each episode emits per-subgoal plots (`epNNNN_sgMM.png`) plus one overlay
    figure (`epNNNN_overlay.png`) with all subgoal progress curves together.
    """
    out, plots, samples, vlm = _prepare(num_samples, model_name, out_path, cache_dir, max_frames, plots_dir)

    all_subgoals: list[SubgoalResult] = []
    with out.open("w", encoding="utf-8") as fh:
        for i, s in enumerate(samples):
            subgoals = split_subgoals(s.caption)
            print(f"[run] {i + 1}/{len(samples)} ep={s.episode_index} frames={len(s.frames)} subgoals={len(subgoals)}")
            sg_results: list[SubgoalResult] = []
            for j, sg in enumerate(subgoals):
                print(f"    subgoal {j + 1}/{len(subgoals)} :: {sg[:70]!r}")
                fields = _score_fields(vlm, s.frames, sg, s.fps, num_prefixes)
                sr = SubgoalResult(episode_index=s.episode_index, subgoal_index=j, caption=sg, num_frames=len(s.frames), **fields)
                if plots is not None and sr.error is None:
                    _try_plot(plot_sample, sr, s.frames, plots / f"ep{s.episode_index:04d}_sg{j:02d}.png")
                sg_results.append(sr)
            all_subgoals.extend(sg_results)

            ep_res = EpisodeResult(
                episode_index=s.episode_index,
                task_index=s.task_index,
                full_caption=s.caption,
                num_frames=len(s.frames),
                num_subgoals=len(subgoals),
                subgoals=sg_results,
            )
            if plots is not None:
                _try_plot(plot_episode_overlay, ep_res, plots / f"ep{s.episode_index:04d}_overlay.png", what="overlay plot")
            _write_record(fh, ep_res)

    _summarize(all_subgoals, model_name, out)


def _summarize(results: list[SampleResult] | list[SubgoalResult], model_name: str, out: Path) -> None:
    ok = [r for r in results if r.error is None]
    summary = {
        "model": model_name,
        "dataset": EGO4D_REPO,
        "num_samples": len(results),
        "num_valid": len(ok),
        "num_errors": len(results) - len(ok),
        "timestamp": datetime.now().isoformat(),
        "reward_mean_stats": summary_stats(r.reward_mean for r in ok),
        "answer_prob_stats": summary_stats(r.answer_token_prob for r in ok),
        "voc_stats": summary_stats(r.voc for r in ok),
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
    p.add_argument(
        "--split-subgoals",
        action="store_true",
        help="split each episode caption into its sub-actions and score/plot each subgoal independently over the full clip",
    )
    args = p.parse_args()
    runner = run_subgoals if args.split_subgoals else run
    runner(
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
