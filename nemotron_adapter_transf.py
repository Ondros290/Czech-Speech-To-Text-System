"""Nemotron 3.5 ASR (cache-aware FastConformer-RNNT) adapter for bench.py.

Built on the Hugging Face Transformers streaming API from the model card /
docs (needs transformers >= 5.13, marked "experimental" there). The API is
PULL based: model.generate() runs in its own thread and pulls feature chunks
from a generator. The benchmark is PUSH based: it calls process(chunk) and
times it. This adapter bridges the two:

    process(chunk) -> buffer audio -> for every complete native window:
        extract features (main thread, timed)
        -> queue -> generator -> generate() (worker thread)
        <- ack queue: generator was asked for the NEXT chunk, i.e. the
           previous chunk is fully decoded
    So process() returns only when the model is done with the audio it was
    given, and the time of that call is the compute time of that step.

Native chunk = streaming latency preset (80/320/560/1120 ms ...). The feed
chunk of the benchmark (--chunk-ms) is independent: use a feed chunk that is
<= the native chunk (e.g. 80 ms). Calls that do not complete a native chunk
just buffer and return the old text; that waiting is real latency.

Differences from the documented example (all deliberate):
  * the example drops the final partial chunk; here the tail is padded with
    silence and decoded on is_last, so the last words are not lost;
  * a window is processed when it is COMPLETE (end <= buffered samples); the
    example uses '<', which would add a whole extra feed chunk of delay;
  * text is decoded from the collected token ids on every call instead of
    through TextIteratorStreamer, which holds back the last, unfinished word
    until the next space and would inflate word latency.

Not run against the real model by me: check with the smoke test first
  python nemotron_adapter.py some_czech.wav --latency-ms 560 --language cs-CZ
"""
from __future__ import annotations

import contextlib
import queue
import threading

import numpy as np

from bench import StreamingASR

ACK_TIMEOUT_S = 300.0


class _IdCollector:
    """Minimal HF 'streamer': keeps every token id generate() emits."""

    def __init__(self):
        self.ids: list[int] = []

    def put(self, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        self.ids.extend(np.asarray(value).reshape(-1).tolist())

    def end(self):
        pass


class NemotronStreamingAdapter(StreamingASR):
    def __init__(self, model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
                 device: str = "cpu", latency_ms: int = 560, language: str = "cs-CZ",
                 dtype: str = "float32", _processor=None, _model=None):
        self.device, self.language = device, language
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForRNNT.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).to(device).eval()
        
        if _model is None:
           
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForRNNT.from_pretrained(
                model_id, dtype=getattr(torch, dtype)).to(device).eval()
        else:  # injected objects (used by the unit tests)
            self.processor, self.model = _processor, _model

        table = self.processor.supported_streaming_latencies_ms  # {lookahead_tokens: ms}
        by_ms = {ms: n for n, ms in table.items()}
        if latency_ms not in by_ms:
            raise ValueError(f"latency {latency_ms} ms not supported; choose from {sorted(by_ms)}")
        self.lookahead = by_ms[latency_ms]
        self.processor.set_num_lookahead_tokens(self.lookahead)  # re-derives chunk sizes
        self.name = f"nemotron-3.5-{latency_ms}ms"

        p, fe = self.processor, self.processor.feature_extractor
        self.sr, self.hop, self.n_fft = fe.sampling_rate, fe.hop_length, fe.n_fft
        self.n_first = p.num_samples_first_audio_chunk
        self.n_per = p.num_samples_per_audio_chunk
        self.frames_first = p.num_mel_frames_first_audio_chunk
        self.frames_per = p.num_mel_frames_per_audio_chunk
        self.tokenizer = p.tokenizer
        cfg = getattr(self.model, "config", None)
        self._drop = {getattr(cfg, "blank_token_id", None), getattr(cfg, "pad_token_id", None)} - {None}
        self._thread = None
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self._shutdown()
        self.audio = np.zeros(0, dtype=np.float32)
        self.started = False
        self.mel_idx = 0
        self.streamer = _IdCollector()
        self._q: queue.Queue = queue.Queue()
        self._ack: queue.Queue = queue.Queue()
        self._exc = None
        self._thread = None
        self._text, self._n_ids = "", 0

    def _shutdown(self):
        t = self._thread
        if t is not None and t.is_alive():
            self._q.put(None)
            t.join(timeout=10)

    def sync(self):
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()

    # ----------------------------------------------------------------- worker
    def _gen(self, first_feats):
        yield first_feats
        self._ack.put(True)          # generate() asked for more => chunk done
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item
            self._ack.put(True)

    def _worker(self, kwargs, first_feats):
        try:
            try:
                import torch
                ctx = torch.inference_mode()
            except ImportError:
                ctx = contextlib.nullcontext()
            with ctx:
                self.model.generate(**kwargs, input_features=self._gen(first_feats),
                                    streamer=self.streamer)
        except BaseException as e:   # surface errors instead of deadlocking
            self._exc = e
            self._ack.put(None)

    def _wait_ack(self):
        try:
            self._ack.get(timeout=ACK_TIMEOUT_S)
        except queue.Empty:
            raise TimeoutError("Nemotron generate() thread stalled") from None
        if self._exc is not None:
            raise RuntimeError("Nemotron generate() thread failed") from self._exc

    # ------------------------------------------------------------- processing
    def _start(self):
        audio = self.audio[: self.n_first]
        if len(audio) < self.n_first:                       # utterance shorter than 1st chunk
            audio = np.pad(audio, (0, self.n_first - len(audio)))
        first = self.processor(audio, sampling_rate=self.sr, is_streaming=True,
                               is_first_audio_chunk=True, language=self.language,
                               return_tensors="pt")
        first = first.to(self.device, dtype=self.model.dtype)
        first_feats = first["input_features"][:, : self.frames_first, :]
        kwargs = {k: v for k, v in first.items() if k != "input_features"}
        kwargs.setdefault("num_lookahead_tokens", self.lookahead)
        self.mel_idx = self.frames_first
        self.started = True
        self._thread = threading.Thread(target=self._worker, args=(kwargs, first_feats),
                                        daemon=True)
        self._thread.start()
        self._wait_ack()

    def _send(self, window):
        inputs = self.processor(window, sampling_rate=self.sr, is_streaming=True,
                                is_first_audio_chunk=False, language=self.language,
                                return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        self._q.put(inputs["input_features"])
        self._wait_ack()
        self.mel_idx += self.frames_per

    def _feed_ready(self, is_last):
        while True:
            start = self.mel_idx * self.hop - self.n_fft // 2
            end = start + self.n_per
            if end <= len(self.audio):
                window = self.audio[start:end]
            elif is_last and self.mel_idx * self.hop < len(self.audio):
                window = np.pad(self.audio[start:], (0, end - len(self.audio)))  # tail + silence
            else:
                break
            self._send(window)

    def _text_now(self):
        if len(self.streamer.ids) != self._n_ids:
            ids = [i for i in self.streamer.ids if i not in self._drop]
            self._text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            self._n_ids = len(self.streamer.ids)
        return self._text

    def process(self, chunk, is_last):
        if len(chunk):
            self.audio = np.concatenate([self.audio, chunk.astype(np.float32, copy=False)])
        if not self.started and len(self.audio) and (len(self.audio) >= self.n_first or is_last):
            self._start()
        if self.started:
            self._feed_ready(is_last)
            if is_last:
                self._q.put(None)                       # end of stream
                self._thread.join(timeout=ACK_TIMEOUT_S)
                if self._thread.is_alive():
                    raise TimeoutError("Nemotron generate() did not finish")
                if self._exc is not None:
                    raise RuntimeError("Nemotron generate() thread failed") from self._exc
        return self._text_now()


# ------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import argparse
    import time

    from bench import SR, load_audio, normalize, run_stream, align

    ap = argparse.ArgumentParser(description="Smoke test with the real model")
    ap.add_argument("audio")
    ap.add_argument("--latency-ms", type=int, default=560)
    ap.add_argument("--language", default="cs-CZ")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feed-ms", type=int, default=80)
    a = ap.parse_args()

    ad = NemotronStreamingAdapter(device=a.device, latency_ms=a.latency_ms, language=a.language)
    audio = load_audio(a.audio)
    for _ in range(2):  # run twice: 2nd run must give the same text (reset() works)
        steps, dur = run_stream(ad, audio, a.feed_ms)
    print(f"audio {dur:.1f}s, {len(steps)} feed chunks")
    last = ""
    for k, s in enumerate(steps):
        if s.text != last:
            print(f"  t={s.finish:6.2f}s (compute {1000 * s.compute:6.1f} ms): {s.text}")
            last = s.text
    print("final latency: %.0f ms, RTF %.3f" % (1000 * (steps[-1].finish - dur),
                                                sum(s.compute for s in steps) / dur))
    print("FINAL:", steps[-1].text)

    # offline reference with the same model (full context) for a rough comparison
    import torch
    inp = ad.processor(audio, sampling_rate=SR, language=a.language)
    inp = inp.to(ad.device, dtype=ad.model.dtype)
    with torch.inference_mode():
        out = ad.model.generate(**inp, return_dict_in_generate=True)
    ref = ad.processor.decode(out.sequences, skip_special_tokens=True)
    ref = ref[0] if isinstance(ref, list) else ref
    errs, _ = align(normalize(ref).split(), normalize(steps[-1].text).split())
    print("OFFLINE:", ref)
    print(f"streaming vs offline word differences: {errs} of {len(normalize(ref).split())}")
