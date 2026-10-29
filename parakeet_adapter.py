"""Parakeet-TDT-0.6B-v3 (FastConformer encoder + TDT decoder) adapter for bench.py.

Parakeet-TDT is an OFFLINE model: it transcribes a complete audio buffer. To
benchmark it in the streaming harness it is wrapped by the SAME policy as
Whisper and wav2vec 2.0 (ReDecodeAdapter: re-decode the growing buffer every
--step-ms, commit the words on which two consecutive decodes agree, final
full decode on the last chunk). What you measure is model + that policy, not a
native streaming system, and word latency is dominated by --step-ms.

v3 is multilingual with automatic language detection per call and has no
language argument. On short partial buffers the detected language can flip
(e.g. to Slovak or Polish); the agreement rule hides most of that, but check
the transcripts (bench.py --out ..._hyp.jsonl) for non-Czech output.

Calls the documented NeMo API: ASRModel.from_pretrained(...) and
model.transcribe([numpy_array], batch_size=1) which returns objects with .text.
transcribe() builds a data loader on every call, so each step carries a fixed
overhead that a direct encoder/decoder call would not have. The smoke test
measures it (python parakeet_adapter.py file.wav); report it in the thesis.
"""
from __future__ import annotations

import time

import numpy as np

from bench import SR

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
MIN_SAMPLES = SR // 10          # shorter buffers are zero-padded (0.1 s)


def _text_of(item) -> str:
    if isinstance(item, str):
        return item
    if hasattr(item, "text"):
        return item.text
    if isinstance(item, (list, tuple)) and item:
        return _text_of(item[0])
    return str(item)


def first_text(out) -> str:
    """NeMo versions differ: list[Hypothesis | str], or a (best, all) tuple."""
    if isinstance(out, tuple):
        out = out[0]
    return _text_of(out[0])


class Transcriber:
    """audio (float32, 16 kHz) -> text, using the documented transcribe() API."""

    CANDIDATE_KWARGS = (
        {"batch_size": 1, "verbose": False, "num_workers": 0},
        {"batch_size": 1, "verbose": False},
        {"batch_size": 1},
    )

    def __init__(self, model):
        self.model = model
        self.kwargs = self._probe()          # done here, so it is never inside a timed call

    def _probe(self):
        silence = np.zeros(SR, dtype=np.float32)
        for kw in self.CANDIDATE_KWARGS:     # keep the first signature this NeMo version accepts
            try:
                self.model.transcribe([silence], **kw)
                return kw
            except TypeError:
                continue
        raise RuntimeError("model.transcribe() rejected every argument combination tried")

    def __call__(self, audio: np.ndarray) -> str:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if len(audio) < MIN_SAMPLES:
            audio = np.pad(audio, (0, MIN_SAMPLES - len(audio)))
        return first_text(self.model.transcribe([audio], **self.kwargs))


def load_parakeet(model_id: str = DEFAULT_MODEL, device: str = "cuda"):
    import nemo.collections.asr as nemo_asr
    try:                                      # silence NeMo's INFO logging
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(30)
    except Exception:
        pass
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    return model.to(device).eval()


def build_parakeet(model_id: str = DEFAULT_MODEL, device: str = "cuda",
                   step_ms: int = 1000, _model=None):
    from adapters import ReDecodeAdapter, _cuda_sync
    model = _model if _model is not None else load_parakeet(model_id, device)
    decode = Transcriber(model)
    return ReDecodeAdapter(decode, model_id.split("/")[-1], step_ms, _cuda_sync(device))


# ------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import argparse
    import statistics

    from bench import load_audio, run_stream

    ap = argparse.ArgumentParser(description="Smoke test with the real model")
    ap.add_argument("audio")
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--step-ms", type=int, default=1000)
    ap.add_argument("--feed-ms", type=int, default=160)
    a = ap.parse_args()

    ad = build_parakeet(a.model_id, a.device, a.step_ms)
    decode, audio = ad.decode_fn, load_audio(a.audio)

    quiet = np.zeros(SR // 2, dtype=np.float32)   # fixed per-call overhead of transcribe()
    for _ in range(3):
        decode(quiet)
    times = []
    for _ in range(15):
        t0 = time.perf_counter(); decode(quiet); ad.sync(); times.append(time.perf_counter() - t0)
    print(f"transcribe() on 0.5 s of silence: median {1000 * statistics.median(times):.0f} ms "
          f"(fixed overhead floor per re-decode step)")

    for _ in range(2):                            # 2nd run must give the same text
        steps, dur = run_stream(ad, audio, a.feed_ms)
    last = ""
    for s in steps:
        if s.text != last:
            print(f"  t={s.finish:6.2f}s (compute {1000 * s.compute:6.1f} ms): {s.text}")
            last = s.text
    off = decode(audio)
    print("final latency: %.0f ms, RTF %.3f" % (1000 * (steps[-1].finish - dur),
                                                sum(s.compute for s in steps) / dur))
    print("FINAL  :", steps[-1].text)
    print("OFFLINE:", off)
    print("final == offline decode of the whole file:", steps[-1].text == off)
