"""Prepare a Czech test set from ParCzech4Speech (LINDAT release): one script,
one filtering strategy.

  plan   choose which AudioPSP recordings you need (metadata only, seconds)
  build  cut the segments out of the recordings you have and write manifest.jsonl

Both commands call the same selection code (is_eligible / plan_recordings /
sampling_order). `plan` stores EVERY selection parameter in <out>/plan.json and
`build` reads them from there; build has no selection flags of its own, so the
two steps cannot disagree.

Typical workflow
  1. python prepare_testset.py plan  --segmented sentence_segmented.parquet \
         --split test --hours 3 --seed 1 --audio-root data/AudioPSP --out data/test_wer
        -> data/test_wer/plan.json, needed_files.txt (all planned recordings),
           missing_files.txt (planned recordings not yet on disk)
     (add --txt-only to write just missing_files.txt, e.g. to start the download first)
  2. download the missing recordings, e.g.
         python fetch_from_tars.py --needed data/test_wer/missing_files.txt \
             --out-root data/AudioPSP URL1 URL2 ...
  3. python prepare_testset.py build --segmented sentence_segmented.parquet \
         --raw raw_alignment_zstd.parquet --audio-root data/AudioPSP --out data/test_wer
        -> wav/ , manifest.jsonl , selection.json

build samples ONLY from recordings that exist under --audio-root, so it works
with whatever you have on disk (planned or not) and reports how much of the
plan is covered. Use a separate --out folder per split/purpose (dev, wer,
latency); each folder holds its own plan.json.

Reference text = true_text (official transcript). rec_text / wav2vec_rec are
Whisper / wav2vec 2.0 outputs and are never used as references. Word times
come from the raw alignment (end time of the recognised word each official
word was aligned to); use --no-words to skip them (no per-word latency).
Requires: pandas, pyarrow, ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench import normalize  # same normalizer as the benchmark

GAP = "<gap>"
BAD_CHARS = re.compile(r"[0-9%&@§]")  # digits/symbols: format mismatch + no timestamps
SEG_COLS = ["true_text", "speaker", "dur", "n_numbers", "speaker_text_cnt", "split",
            "filePath", "start", "end", "start_token_id", "end_token_id", "seg_edit_dist"]
RAW_COLS = ["true_word", "rec_word", "start", "end", "token_id", "is_token_word", "filePath"]


# ============================================================ shared helpers
def _nan(x) -> bool:
    return x is None or x != x


def clean_rel(p) -> str:
    p = str(p).replace("\\", "/")
    while p.startswith("./") or p.startswith("/"):
        p = p[2:] if p.startswith("./") else p[1:]
    return p


def load_segments(path, split):
    import pandas as pd
    seg = pd.read_parquet(path, columns=SEG_COLS)
    seg = seg[seg["split"] == split].reset_index(drop=True)
    meta = seg.to_dict("records")
    for r in meta:
        r["n_words"] = len(normalize(r["true_text"]).split())
    return meta


# ================================================== THE selection strategy
def is_eligible(r, p) -> bool:
    """The one place that decides whether a segment may be in the test set."""
    if not (p["min_dur"] <= r["dur"] <= p["max_dur"]) or r["n_words"] < p["min_words"]:
        return False
    if p["dedupe"] and r["speaker_text_cnt"] != 1:
        return False
    if not p["allow_numbers"] and (r["n_numbers"] > 0 or BAD_CHARS.search(r["true_text"])):
        return False
    return True


def plan_recordings(meta, p, exists=lambda fp: False):
    """Pick recordings holding ~hours*margin of eligible audio. Recordings that
    already exist on disk come first, then the rest (dense or random)."""
    per = defaultdict(lambda: {"sec": 0.0, "n": 0, "spk": set(), "end": 0.0})
    for r in meta:
        d = per[r["filePath"]]
        d["end"] = max(d["end"], r["end"])
        if is_eligible(r, p):
            d["sec"] += r["dur"]
            d["n"] += 1
            d["spk"].add(r["speaker"])
    files = [f for f, d in per.items() if d["n"] > 0]
    if p["strategy"] == "dense":      # most eligible audio per minute of recording first
        files.sort(key=lambda f: (not exists(f), -per[f]["sec"] / max(per[f]["end"], 1.0)))
    else:
        random.Random(p["seed"]).shuffle(files)
        files.sort(key=lambda f: not exists(f))
    chosen, cum = [], 0.0
    for f in files:
        if cum >= p["hours"] * 3600 * p["margin"]:
            break
        chosen.append(f)
        cum += per[f]["sec"]
    return chosen, per, files


def sampling_order(meta, p, allowed):
    """Eligible segments from `allowed` recordings, interleaved over speakers (seeded)."""
    rng = random.Random(p["seed"])
    by_spk = defaultdict(list)
    for i, r in enumerate(meta):
        if r["filePath"] in allowed and is_eligible(r, p):
            by_spk[r["speaker"]].append(i)
    for v in by_spk.values():
        rng.shuffle(v)
    speakers = sorted(by_spk)
    rng.shuffle(speakers)
    order = []
    while any(by_spk[s] for s in speakers):
        for s in speakers:
            if by_spk[s]:
                order.append(by_spk[s].pop())
    return order


def take_hours(order, meta, hours):
    out, total = [], 0.0
    for i in order:
        if total >= hours * 3600:
            break
        out.append(i)
        total += meta[i]["dur"]
    return out


# ======================================================== build-only helpers
def words_for_segment(rows, cut_start, true_text):
    """rows: raw-alignment rows from start to end token. -> ([{"w","end"}], None) or (None, reason)."""
    words = []
    for r in rows:
        if not r["is_token_word"]:
            continue
        tw, rw = r["true_word"], r["rec_word"]
        if tw == GAP:               # recognised word absent from the official text
            continue
        if rw == GAP or _nan(r["start"]) or _nan(r["end"]):
            return None, "word without timestamp"
        words.append({"w": tw, "end": round(float(r["end"]) - cut_start, 3)})
    if " ".join(normalize(w["w"]) for w in words).split() != normalize(true_text).split():
        return None, "words do not reproduce true_text"
    return words, None


def cut_audio(src, cut_start, dur, dst):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{cut_start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def existing_recordings(root, meta):
    return {fp for fp in {r["filePath"] for r in meta} if (root / clean_rel(fp)).is_file()}


# ================================================================== commands
def cmd_plan(args):
    p = {k: getattr(args, k) for k in ("split", "hours", "seed", "min_dur", "max_dur",
                                       "min_words", "margin", "strategy")}
    p["allow_numbers"], p["dedupe"] = args.allow_numbers, not args.no_dedupe
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = load_segments(args.segmented, p["split"])
    exists = lambda fp: False
    root = None
    if args.audio_root:
        root = Path(args.audio_root).resolve()
        have = existing_recordings(root, meta)
        exists = lambda fp: fp in have
        print(f"{len(have)} recordings of split {p['split']!r} already exist under {root}")

    chosen, per, eligible_files = plan_recordings(meta, p, exists)
    missing = [f for f in chosen if not exists(f)]
    got_h = sum(per[f]["sec"] for f in chosen) / 3600
    all_h = sum(per[f]["sec"] for f in eligible_files) / 3600
    spk = set().union(*(per[f]["spk"] for f in chosen)) if chosen else set()
    spk_all = set().union(*(per[f]["spk"] for f in eligible_files)) if eligible_files else set()
    rec_h = sum(per[f]["end"] for f in missing) / 3600

    (out / "missing_files.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
    if not args.txt_only:
        (out / "plan.json").write_text(json.dumps({
            "params": p, "segmented": str(args.segmented), "files": chosen,
            "stats": {"eligible_hours_in_plan": got_h, "eligible_hours_total": all_h,
                      "speakers_in_plan": len(spk), "speakers_total": len(spk_all)}},
            indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "needed_files.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")

    print(f"split {p['split']!r}: {len(eligible_files)} recordings hold eligible segments "
          f"({all_h:.1f} h eligible audio, {len(spk_all)} speakers)")
    print(f"plan ({p['strategy']}): {len(chosen)} recordings -> {got_h:.2f} h eligible audio, "
          f"{len(spk)} speakers")
    print(f"{len(chosen) - len(missing)} already on disk, {len(missing)} still to download "
          f"(~{rec_h:.1f} h of recordings; multiply by bytes/hour of your format)")
    if args.txt_only:
        print(f"wrote {out / 'missing_files.txt'} ({len(missing)} recordings) and nothing else")
        if (out / "plan.json").exists():
            print("note: an existing plan.json in this folder was left untouched and may not "
                  "match these parameters")
        print("to run 'build' later, run 'plan' again WITHOUT --txt-only (same parameters)")
    else:
        print(f"wrote {out}/plan.json, needed_files.txt, missing_files.txt")


def cmd_build(args):
    out = Path(args.out)
    plan_path = Path(args.plan) if args.plan else out / "plan.json"
    if not plan_path.is_file():
        raise SystemExit(f"{plan_path} not found: run 'plan' first (same --out folder)")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    p = plan["params"]
    if not args.no_words and not args.raw:
        raise SystemExit("--raw is required unless --no-words is given")
    print(f"plan: split={p['split']} hours={p['hours']} seed={p['seed']} "
          f"(parameters read from {plan_path})")

    import pandas as pd

    root = Path(args.audio_root).resolve()
    meta = load_segments(args.segmented or plan["segmented"], p["split"])
    have = existing_recordings(root, meta)
    planned = set(plan["files"])
    print(f"{len(have)} recordings of the split exist under {root}; "
          f"{len(planned & have)} of {len(planned)} planned recordings present")
    if not have:
        raise SystemExit("no recordings found; check --audio-root (relative like data/AudioPSP, "
                         "or a full path with drive letter; '/data' means the drive root on Windows)")

    order = sampling_order(meta, p, have)
    avail_h = sum(meta[i]["dur"] for i in order) / 3600
    print(f"{len(order)} eligible segments on disk ({avail_h:.2f} h); target {p['hours']} h")
    if avail_h < p["hours"]:
        print("WARNING: less eligible audio on disk than the target; download more "
              "(see missing_files.txt) or lower --hours in plan")
    cand = take_hours(order, meta, p["hours"] * 1.2)   # extra: some fail the word check

    words_by_idx, reasons = {}, Counter()
    if args.no_words:
        words_by_idx = {i: None for i in cand}
    else:
        by_file = defaultdict(list)
        for i in cand:
            by_file[meta[i]["filePath"]].append(i)
        for n, (fp, idxs) in enumerate(by_file.items(), 1):
            raw = pd.read_parquet(args.raw, columns=RAW_COLS,
                                  filters=[("filePath", "==", fp)]).reset_index(drop=True)
            pos = {}
            for k, tid in enumerate(raw["token_id"]):
                if not _nan(tid) and tid not in pos:
                    pos[tid] = k
            recs = raw.to_dict("records")
            for i in idxs:
                m = meta[i]
                i0, i1 = pos.get(m["start_token_id"]), pos.get(m["end_token_id"])
                if i0 is None or i1 is None or i1 < i0:
                    reasons["token ids not found"] += 1
                    continue
                w, why = words_for_segment(recs[i0:i1 + 1], max(0.0, m["start"] - args.pad),
                                           m["true_text"])
                if w is None:
                    reasons[why] += 1
                else:
                    words_by_idx[i] = w
            print(f"  raw alignment: file {n}/{len(by_file)}")

    kept = take_hours([i for i in cand if i in words_by_idx], meta, p["hours"])
    total = sum(meta[i]["dur"] for i in kept)
    print(f"kept {len(kept)} segments, {total / 3600:.2f} h; dropped: {dict(reasons)}")

    (out / "wav").mkdir(parents=True, exist_ok=True)
    speakers = set()
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for i in kept:
            m = meta[i]
            cut_start = max(0.0, m["start"] - args.pad)
            cut_dur = (m["end"] + args.pad) - cut_start
            uid = f"{p['split']}_{i:07d}"
            wav = (out / "wav" / f"{uid}.wav").resolve()
            cut_audio(root / clean_rel(m["filePath"]), cut_start, cut_dur, wav)
            rec = {"id": uid, "audio": str(wav), "text": m["true_text"],
                   "speaker": m["speaker"], "dur": round(cut_dur, 3),
                   "seg_edit_dist": m["seg_edit_dist"]}
            if words_by_idx[i] is not None:
                rec["words"] = words_by_idx[i]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            speakers.add(m["speaker"])

    (out / "selection.json").write_text(json.dumps({
        "params": p, "pad": args.pad, "n_segments": len(kept), "hours": total / 3600,
        "n_speakers": len(speakers), "n_recordings": len({meta[i]["filePath"] for i in kept}),
        "n_words": sum(meta[i]["n_words"] for i in kept),
        "planned_recordings_present": len(planned & have), "planned_recordings": len(planned),
        "dropped": dict(reasons)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out / 'manifest.jsonl'} ({len(speakers)} speakers)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="choose recordings to download (writes plan.json)")
    pl.add_argument("--segmented", required=True)
    pl.add_argument("--split", default="test", help="value of the 'split' column")
    pl.add_argument("--hours", type=float, default=3.0, help="target hours of eligible audio")
    pl.add_argument("--seed", type=int, default=1)
    pl.add_argument("--min-dur", type=float, default=2.0)
    pl.add_argument("--max-dur", type=float, default=30.0)
    pl.add_argument("--min-words", type=int, default=3)
    pl.add_argument("--allow-numbers", action="store_true")
    pl.add_argument("--no-dedupe", action="store_true")
    pl.add_argument("--strategy", choices=["dense", "random"], default="dense")
    pl.add_argument("--margin", type=float, default=1.5,
                    help="plan this many times the target (some segments get dropped)")
    pl.add_argument("--audio-root", default=None,
                    help="optional: recordings already there are used first and not re-downloaded")
    pl.add_argument("--txt-only", action="store_true",
                    help="write ONLY missing_files.txt (the list for fetch_from_tars.py); "
                         "no plan.json, so 'build' cannot be run from this output")
    pl.add_argument("--out", required=True)
    pl.set_defaults(func=cmd_plan)

    bd = sub.add_parser("build", help="cut audio + write manifest.jsonl (reads plan.json)")
    bd.add_argument("--segmented", default=None, help="default: path stored in plan.json")
    bd.add_argument("--raw", default=None, help="raw alignment parquet (for word latency)")
    bd.add_argument("--audio-root", required=True)
    bd.add_argument("--pad", type=float, default=0.1,
                    help="seconds of audio kept before/after a segment (avoid clipped onsets)")
    bd.add_argument("--no-words", action="store_true")
    bd.add_argument("--plan", default=None, help="default: <out>/plan.json")
    bd.add_argument("--out", required=True)
    bd.set_defaults(func=cmd_build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
