"""Streaming ASR latency + WER benchmark with a simulated real-time clock.

Every model is wrapped in the same StreamingASR interface and fed the same
audio in fixed-size chunks. Chunk k "arrives" when its last sample would have
been recorded (virtual time), a single worker processes chunks in order, and
the measured wall-clock compute time of each call advances the virtual clock.
If a model is slower than real time, a queue builds up and latency grows,
exactly as it would in a live system. No sleeping is needed, so runs are fast
and reproducible.

Manifest (JSONL), one utterance per line:
  {"id": "u1", "audio": "path.wav", "text": "reference transcript",
   "words": [{"w": "slovo", "end": 1.23}, ...]}     # "words" is optional
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import statistics
import time
import unicodedata
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000


# --------------------------------------------------------------------------
# Adapter interface
# --------------------------------------------------------------------------
class StreamingASR(ABC):
    name = "base"

    @abstractmethod
    def reset(self) -> None:
        """Clear all per-utterance state (called before every stream)."""

    @abstractmethod
    def process(self, chunk: np.ndarray, is_last: bool) -> str:
        """Feed one float32 mono 16 kHz chunk.

        Return the CUMULATIVE hypothesis so far (committed + partial text).
        On is_last=True the adapter must flush and return the final text.
        """

    


# --------------------------------------------------------------------------
# Text normalisation + alignment
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, NFC, strip punctuation, keep Czech diacritics.

    NOTE: numbers are NOT converted (e.g. "21" vs "dvacet jedna")."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def align(ref: list[str], hyp: list[str]):
    """Levenshtein distance + list of exactly matching (ref_idx, hyp_idx).   
        Args:        
    
        Returns:
            _type_: edit distance - the minimum number of substitutions, deletions, and insertions needed to turn the hypothesis into the reference
            matched pairs - he positions of words that are identical in both lists after the best alignment
        """
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    i, j, matches = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] == hyp[j - 1]:
                matches.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return int(d[n, m]), matches[::-1]


# --------------------------------------------------------------------------
# Streaming runner (virtual clock)
# --------------------------------------------------------------------------
@dataclass
class Step:
    arrival: float   # when the chunk's last sample was available (s)
    start: float     # when processing started (>= arrival, queueing)
    finish: float    # when the hypothesis was emitted (s, virtual time)
    compute: float   # measured wall-clock compute for this call (s)
    text: str


def run_stream(asr: StreamingASR, audio: np.ndarray, chunk_ms: int):
    asr.reset()
    n = int(SR * chunk_ms / 1000) # number of samples per chunk
    n_chunks = max(1, -(-len(audio) // n)) # number of chunks per audio
    steps: list[Step] = []
    free_at = 0.0
    gc.collect()
    gc.disable()  # avoid GC pauses landing inside a timed call
    try:
        for k in range(n_chunks):
            chunk = audio[k * n:(k + 1) * n] # take the current chunk from the audio
            arrival = min((k + 1) * n, len(audio)) / SR # time in seconds of the chunk end in the audio
            t0 = time.perf_counter() 
            text = asr.process(chunk, k == n_chunks - 1)  # get the hypothetic text from the ASR model
            # asr.sync()
            compute = time.perf_counter() - t0 # save compute time
            start = max(arrival, free_at) 
            free_at = start + compute 
            steps.append(Step(arrival, start, free_at, compute, text))
    finally:
        gc.enable()
    return steps, len(audio) / SR


def stable_word_times(steps: list[Step]):
    """For each word position, the time it last changed = when it became stable.

    Handles models that revise partial hypotheses: a word only counts as
    emitted once it never changes again.
    """
    words: list[str] = []
    stable: list[float] = []
    for s in steps:
        new = normalize(s.text).split()
        for i, w in enumerate(new):
            if i >= len(words) or words[i] != w:
                if i < len(stable):
                    stable[i] = s.finish
                else:
                    stable.append(s.finish)
        stable = stable[:len(new)]
        words = new
    return words, stable


# --------------------------------------------------------------------------
# Per-utterance metrics
# --------------------------------------------------------------------------
def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def evaluate(utt: dict, steps: list[Step], dur: float):
    
    hyp, stable = stable_word_times(steps) # time the hypothesis became stable (did not change)

    if utt.get("words"):
        pairs = [(normalize(w["w"]), float(w["end"])) for w in utt["words"]]
        pairs = [(w, e) for w, e in pairs if w]
        ref = [w for w, _ in pairs]
        ref_end = [e for _, e in pairs]
    else:
        ref = normalize(utt["text"]).split()
        ref_end = None

    errs, matches = align(ref, hyp)
    computes = [s.compute for s in steps] # processing time
    first_emit = next((s.finish for s in steps if s.text.strip()), float("nan"))

    row = {
        "id": utt["id"],
        "audio_s": dur,
        "n_chunks": len(steps),
        "ref_words": len(ref),
        "errors": errs,
        "rtf": sum(computes) / dur, # real time factor
        "chunk_compute_p50_ms": 1000 * pct(computes, 50), # per chunk compute
        "chunk_compute_p95_ms": 1000 * pct(computes, 95),
        "chunk_compute_max_ms": 1000 * max(computes),
        "queue_delay_max_ms": 1000 * max(s.start - s.arrival for s in steps),
        # time from the last audio sample until the final transcript exists
        "final_latency_ms": 1000 * (steps[-1].finish - dur), # time from the last audio sample to the final transcript existing.
        "first_emit_s": first_emit,
    }

    word_lat = []
    if ref_end is not None:
        word_lat = [stable[j] - ref_end[i] for i, j in matches]
        if word_lat:
            row["first_word_latency_ms"] = 1000 * word_lat[0] # latency is words stable time minus the reference word’s end time
            row["word_lat_p50_ms"] = 1000 * pct(word_lat, 50)
            row["word_lat_p90_ms"] = 1000 * pct(word_lat, 90)
    return row, word_lat


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def load_audio(path: str) -> np.ndarray:
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except ImportError:  # stdlib fallback: 16-bit PCM WAV only
        with wave.open(path, "rb") as w:
            sr, ch = w.getframerate(), w.getnchannels()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            x = (x.reshape(-1, ch) / 32768.0).astype(np.float32)
            x = x if ch > 1 else x[:, 0]
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def load_manifest(path: str, max_utts: int | None): # file with transcipt, information about timing and links to audio files
    utts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            u = json.loads(line)
            u["_audio"] = load_audio(u["audio"])  # pre-load: no disk I/O in timing
            utts.append(u)
            if max_utts and len(utts) >= max_utts:
                break
    return utts


def env_info(device: str) -> dict:
    info = {"device": device}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--adapter", required=True, help="see adapters.build()")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-ms", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--step-ms", type=int, default=1000,
                    help="re-decode interval for offline models wrapped for streaming")
    ap.add_argument("--native-ms", type=int, default=560,
                        help="Nemotron streaming latency preset (80/320/560/1120); run one value per --out")
    ap.add_argument("--language", default="cs-CZ", help="language prompt for Nemotron")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3, help="untimed warm-up utterances")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from adapters import build
    asr = build(args.adapter, args) # builds the model according to input arguments
    utts = load_manifest(args.manifest, args.max_utts) # preloads audio
    print(f"{len(utts)} utterances, adapter={asr.name}") 

    rows, pooled = [], {}
    for chunk_ms in args.chunk_ms:
        for u in utts[: args.warmup]:                      # warm-up, discarded
            run_stream(asr, u["_audio"], chunk_ms)
        pool = pooled.setdefault(chunk_ms, [])
        """ if chunk_ms not in pooled:
            pooled[chunk_ms] = []
        pool = pooled[chunk_ms] """
        for rep in range(args.repeats): # repeat multiple times
            for u in utts:
                steps, dur = run_stream(asr, u["_audio"], chunk_ms) # returns steps: hypothesis + timing_info
                row, wl = evaluate(u, steps, dur)  
                row.update(model=asr.name, chunk_ms=chunk_ms, rep=rep)
                rows.append(row)
                pool.extend(wl)
        print(f"  chunk {chunk_ms} ms done")

    # per-utterance CSV
    keys = sorted({k for r in rows for k in r})
    with open(f"{args.out}_utterances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # summary per chunk size
    summary = {"env": env_info(args.device), "model": asr.name, "results": {}}
    for chunk_ms in args.chunk_ms:
        rs = [r for r in rows if r["chunk_ms"] == chunk_ms]
        r0 = [r for r in rs if r["rep"] == 0]               # WER: deterministic, use rep 0
        s = {
            "wer": sum(r["errors"] for r in r0) / max(1, sum(r["ref_words"] for r in r0)),
            "rtf_median": statistics.median(r["rtf"] for r in rs),
            "final_latency_ms_p50": pct([r["final_latency_ms"] for r in rs], 50),
            "final_latency_ms_p95": pct([r["final_latency_ms"] for r in rs], 95),
            "chunk_compute_ms_p50": pct([r["chunk_compute_p50_ms"] for r in rs], 50),
            "chunk_compute_ms_max": max(r["chunk_compute_max_ms"] for r in rs),
            "queue_delay_ms_max": max(r["queue_delay_max_ms"] for r in rs),
        }
        wl = pooled[chunk_ms]
        if wl:
            s["word_latency_ms_p50"] = 1000 * pct(wl, 50)
            s["word_latency_ms_p90"] = 1000 * pct(wl, 90)
        summary["results"][chunk_ms] = s
        print(f"chunk {chunk_ms:>5} ms: " + ", ".join(f"{k}={v:.3f}" for k, v in s.items()))

    Path(f"{args.out}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
