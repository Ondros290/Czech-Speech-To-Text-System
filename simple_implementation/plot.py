import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.figsize": (6.0, 3.8),
})


def read_index(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_splits(root, splits):
    data = {}
    for s in splits:
        p = Path(root) / f"{s}_index.jsonl"
        if p.exists():
            data[s] = read_index(p)
        else:
            print(f"skipping '{s}': {p} not found")
    if not data:
        raise SystemExit("no index files found")
    return data


def per_split(data, fn):
    """Apply fn(record) -> value to every record, per split, as numpy arrays."""
    return {s: np.array([fn(r) for r in recs], dtype=float) for s, recs in data.items()}


def save(fig, out, name):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")

    plt.close(fig)

def hist_overlay(values, xlabel, name, out, title, bins=50, vline=None, vline_label=None,
                 logy=False, xmax_quantile=None, ylabel="Fraction of clips"):
    """Side-by-side histograms of each split, each normalised to sum to 1."""
    allv = np.concatenate(list(values.values()))
    lo = allv.min()
    hi = np.quantile(allv, xmax_quantile) if xmax_quantile else allv.max()

    if vline is not None:
        hi = max(hi, vline * 1.05)

    edges = np.linspace(lo, hi, bins + 1)

    n = len(values)
    fig, axes = plt.subplots(
        1, n,
        figsize=(6.0 * n, 3.8),
        sharex=True,
        sharey=True,
        squeeze=False
    )
    axes = axes[0]

    for ax, (split, v) in zip(axes, values.items()):
        v = v[(v >= lo) & (v <= hi)]
        w = np.full(len(v), 1.0 / max(len(values[split]), 1))

        ax.hist(v, bins=edges, weights=w, edgecolor="white", linewidth=0.4)

        if vline is not None:
            ax.axvline(
                vline,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=vline_label
            )
            ax.legend(frameon=False)

        if logy:
            ax.set_yscale("log")

        ax.set_title(f"{title} - {split} (n={len(values[split])})")
        ax.set_xlabel(xlabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(ylabel)

    fig.tight_layout()
    save(fig, out, name)


def summary(values, unit):
    for s, v in values.items():
        print(f"  {s:<11} n={len(v):>7}  mean {v.mean():7.2f}  median {np.median(v):7.2f}  "
              f"p99 {np.quantile(v, .99):7.2f}  max {v.max():7.2f} {unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--out", default="./figures")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="model output frames per second (25 for 4x downsampling)")
    a = ap.parse_args()
    out = Path(a.out)
    data = load_splits(a.root, a.splits)

    # overall numbers for the dataset-statistics table
    print("hours per split:")
    for s, recs in data.items():
        print(f"  {s:<11} {sum(r['dur'] for r in recs) / 3600:.2f} h, {len(recs)} clips")

    # 1. clip duration
    dur = per_split(data, lambda r: r["dur"])
    print("duration (s):"); summary(dur, "s")
    hist_overlay(dur, "Clip duration (s)", "hist_duration", out, "Durations")

    # 2. characters per second, with the model's output frame rate marked
    cps = per_split(data, lambda r: len(r["norm_text"]) / r["dur"])
    print("chars/sec:"); summary(cps, "chars/s")
    hist_overlay(cps, "Characters per second", "hist_chars_per_second", out, "Number of characters per second")
    for s, v in cps.items():
        print(f"  {s}: {np.mean(v >= a.fps):.2%} of clips have >= {a.fps:g} chars/s "
              f"(CTC cannot align these)")

    # 3. transcript length in characters and in words
    nchar = per_split(data, lambda r: len(r["norm_text"]))
    hist_overlay(nchar, "Transcript length (characters)", "hist_transcript_chars", out, "Transcript length (in characters)")
    nword = per_split(data, lambda r: len(r["norm_text"].split()))
    print("words per clip:"); summary(nword, "words")
    hist_overlay(nword, "Transcript length (words)", "hist_transcript_words", out, "Transcript length (in words)")

    # 4. clips per speaker (histogram over speakers), log y to show the long tail
    per_spk = {s: np.array(list(Counter(r["speaker"] for r in recs).values()), dtype=float)
               for s, recs in data.items()}
    print("clips per speaker:"); summary(per_spk, "clips")
    hist_overlay(per_spk, "Clips per speaker", "hist_clips_per_speaker", out, "Number of clips per speaker",
                 logy=True, ylabel="Fraction of speakers")

    # speaker overlap between splits (should be zero for a clean evaluation)
    names = list(data)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = {r["speaker"] for r in data[names[i]]} & {r["speaker"] for r in data[names[j]]}
            print(f"speaker overlap {names[i]}/{names[j]}: {len(ov)}")


if __name__ == "__main__":
    main()
