"""Adapters that put every model behind the StreamingASR interface.

Two kinds:
  * native streaming  : Vosk, Nemotron 3.5 ASR (cache-aware FastConformer-RNNT)
  * offline + policy  : Whisper, wav2vec 2.0, Parakeet-TDT, wrapped by the SAME
                        re-decode policy (ReDecodeAdapter, LocalAgreement-2)

For offline models you are benchmarking the model AND the streaming policy.
Say so in the thesis, and report the policy parameters (step_ms, agreement).
"""
from __future__ import annotations

import json
import time

import numpy as np

from bench import SR, StreamingASR


# --------------------------------------------------------------------------
# Native streaming: Vosk (Kaldi hybrid)
# --------------------------------------------------------------------------
class VoskAdapter(StreamingASR):
    def __init__(self):
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        self._KaldiRecognizer = KaldiRecognizer
        self.model = Model(lang="cs")
        self.name = "vosk"

    def reset(self):
        self.rec = self._KaldiRecognizer(self.model, SR)
        self.committed: list[str] = []

    def process(self, chunk, is_last):
        pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
        partial = ""
        if self.rec.AcceptWaveform(pcm):  # endpoint detected -> segment finalised
            self.committed.append(json.loads(self.rec.Result()).get("text", ""))
        else:
            partial = json.loads(self.rec.PartialResult()).get("partial", "")
        if is_last:
            self.committed.append(json.loads(self.rec.FinalResult()).get("text", ""))
            partial = ""
        return " ".join(t for t in self.committed + [partial] if t)



# --------------------------------------------------------------------------
# Offline models made "streaming" by one shared policy
# --------------------------------------------------------------------------
class ReDecodeAdapter(StreamingASR):
    """Re-decode the whole buffer every step_ms; commit the prefix on which two
    consecutive decodes agree (LocalAgreement-2).
    On the last chunk the final full decode is used.

    Limitation: the buffer is never trimmed, so per-step cost grows with
    utterance length.
    """

    def __init__(self, decode_fn, name: str, step_ms: int = 1000, sync_fn=None):
        self.decode_fn, self.name = decode_fn, name
        self.step = int(SR * step_ms / 1000)
        self._sync = sync_fn

    def reset(self):
        self.buf = np.zeros(0, dtype=np.float32)
        self.since = 0
        self.prev: list[str] = []
        self.committed: list[str] = []

    def process(self, chunk, is_last):
        self.buf = np.concatenate([self.buf, chunk])
        self.since += len(chunk)
        if not is_last and self.since < self.step:
            return " ".join(self.committed)
        self.since = 0
        hyp = self.decode_fn(self.buf).split()
        if is_last:
            self.committed = hyp
        else:
            n = 0
            while n < min(len(hyp), len(self.prev)) and hyp[n] == self.prev[n]:
                n += 1
            if n > len(self.committed):
                self.committed = hyp[:n]
            self.prev = hyp
        return " ".join(self.committed)

    def sync(self):
        if self._sync:
            self._sync()


def _cuda_sync(device: str):
    if device.startswith("cuda"):
        import torch
        return torch.cuda.synchronize
    return None


def make_faster_whisper(model_size: str, device: str, language: str = "cs"):
    from faster_whisper import WhisperModel
    m = WhisperModel(model_size, device=device,
                     compute_type="float16" if device == "cuda" else "int8")

    def decode(audio: np.ndarray) -> str:
        segs, _ = m.transcribe(audio, language=language, beam_size=1,
                               condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text for s in segs)  # generator consumed inside timing
    return decode 



def make_hf_ctc(model_id: str, device: str):
    """wav2vec 2.0 / XLS-R CTC checkpoint from Hugging Face, GREEDY decoding, no LM.
 
    Uses the feature extractor and tokenizer directly instead of AutoProcessor:
    repositories that ship an n-gram LM load as a ProcessorWithLM whose
    batch_decode expects logits and would apply the LM, which would make
    results depend on the repo instead of on the architecture.
    """
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to(device).eval()
    min_samples = SR // 10          # the conv front end needs a minimum length
 
    @torch.inference_mode()
    def decode(audio: np.ndarray) -> str:
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
        x = fe(audio, sampling_rate=SR, return_tensors="pt").input_values.to(device)
        return tok.batch_decode(model(x).logits.argmax(-1))[0]
    return decode
 
 
def _require(args, adapter: str, what: str, example: str) -> str:
    if not args.model_id:
        raise SystemExit(f"--adapter {adapter} needs --model-id: {what}\n  e.g. --model-id {example}")
    return args.model_id


def make_nemo_offline(model_id: str, device: str):
    """Offline NeMo model, e.g. a Parakeet-TDT checkpoint with Czech support."""
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained(model_id).to(device).eval()

    def decode(audio: np.ndarray) -> str:
        out = m.transcribe([audio], batch_size=1, verbose=False)
        h = out[0]
        return h.text if hasattr(h, "text") else str(h)
    return decode


# --------------------------------------------------------------------------
# Registry used by bench.py
# --------------------------------------------------------------------------
def build(name: str, args) -> StreamingASR:
    dev = args.device
    
    if name == "vosk":
        return VoskAdapter()
    
    if name == "nemotron":
        from nemotron_adapter_transf import NemotronStreamingAdapter
        return NemotronStreamingAdapter(
            args.model_id or "nvidia/nemotron-3.5-asr-streaming-0.6b", dev,
            latency_ms=args.native_ms, language=args.language)
    
    if name == "parakeet":
        from parakeet_adapter import DEFAULT_MODEL, build_parakeet
        return build_parakeet(args.model_id or DEFAULT_MODEL, dev, args.step_ms)

    if name == "whisper":
        size = args.model_id or "large-v3"
        fn = make_faster_whisper(size, dev)
        return ReDecodeAdapter(fn, f"whisper-{args.model_id}", args.step_ms, _cuda_sync(dev))
    
    if name == "w2v2":
        
        mid = _require(args, "w2v2", "a Czech wav2vec 2.0 / XLS-R CTC checkpoint (Hugging Face repo id "
                       "or local folder); there is deliberately no default",
                       "comodoro/wav2vec2-xls-r-300m-cs-250")
        return ReDecodeAdapter(make_hf_ctc(mid, dev), f"wav2vec2-{mid.split('/')[-1]}",
                               args.step_ms, _cuda_sync(dev))
        
    raise ValueError(f"unknown adapter {name!r}")
