import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Value
from itertools import islice
import io
import soundfile as sf
import torchaudio.functional as F

from pathlib import Path
from collections import Counter

from torch.nn.utils.rnn import pad_sequence
import unicodedata
import re
import os
import csv
import json
import argparse

CZECH_DIACRITICS = set("abcdefghijklmnopqrstuvwxyzáčďéěíňóřšťúůýž")
ROOT = Path("./data")
AUDIO_DIR = ROOT / "audio"
TARGET_SR = 16000
VOCAB_PATH = ROOT / "vocab.json"
N_MELS = 80
BLANK_ID = 0

class LogMelFeaturizer:
    def __init__(self, n_fft=400, hop_length=160, sample_rate=TARGET_SR):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=N_MELS)

    def __call__(self, wav):
        x = torch.log(self.mel(wav).squeeze(0) + 1e-6)  # (n_mels, T)
        x = (x - x.mean()) / (x.std() + 1e-5)
        return x.T.contiguous()  
    
class CzechSpeechDS(Dataset):
    def __init__(self, split, featurize, vocab_path = VOCAB_PATH, max_cps = 25):
        self.samples = read_index(index_path(split))
        self.vocab, self.ids = load_vocab(vocab_path)
        self.featurize = featurize
        
        kept, out_of_vocab, too_fast = [], 0, 0
        for s in self.samples:
            text = s["norm_text"]
            if any(c not in self.ids for c in text):
                out_of_vocab += 1
            elif max_cps and len(text) / s["dur"] > max_cps:
                too_fast += 1
            else:
                kept.append(s)
        self.samples = kept
        print(f"[{split}] {len(kept)} clips (dropped {out_of_vocab} out-of-vocab, "
              f"{too_fast} above {max_cps} chars/s)")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        
        sample = self.samples[i]

        audio,_ = sf.read(AUDIO_DIR / sample["path"], dtype="float32")
        
        features = self.featurize(torch.from_numpy(audio).unsqueeze(0))
        
        tokens = []
        for c in sample["norm_text"]:
            tokens.append(self.ids[c])
        tokens = torch.tensor(tokens, dtype=torch.long)
                
        return features, tokens
    
    def decode(self, ids):
        characters = []
        for i in ids:
            char = self.vocab[i]
            characters.append(char)

        result = "".join(characters)
        return result
            
    
def index_path(split):
    return ROOT / f"{split}_index.jsonl"

def read_index(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def load_vocab(path=VOCAB_PATH):
    itos = json.loads(Path(path).read_text(encoding="utf-8"))["vocab"]
    
    stoi = {}
    for i, c in enumerate(itos):
        stoi[c] = i
        
    return itos, stoi

def collate_function(batch): # batch - list of tuples (spectrogram, text)
    # synchronizing length (by padding)
    features, tgts = zip(*batch) # divides into 2 lists - spectrograms and all texts
    feat_lens = torch.tensor([f.shape[0] for f in features]) # lengths of the spectrograms
    tgt_lens = torch.tensor([len(t) for t in tgts]) # lengths of the target texts
    feats = pad_sequence(features, batch_first=True)  # (B, T, 80) Pad with zeros
    tgts = torch.cat(tgts)  # CTC accepts concatenated targets
    return feats, feat_lens, tgts, tgt_lens

def inspect_examples(split, n=50):

    dataset = load_dataset("ufal/parczech4speech-segmented", split=split, streaming=True)

    dataset._info.features["flac"] = Value("binary")

    for i, ex in enumerate(islice(dataset, 50)):
        raw_flac = ex["flac"]
        audio_bytes = raw_flac["bytes"] if isinstance(raw_flac, dict) else raw_flac
        array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        meta = ex["json"]
        print(i, sr, len(array), meta.get("dur"), meta.get("true_text"))

def sample_subset(hours, split, seed, min_dur, max_dur):
    
    # filter segments with numbers
    target_sec = hours * 3600.0
    
    index_file = index_path(split)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    
    records = read_index(index_file) if index_file.exists() else []
    
    done = {r["id"] for r in records}
    total = sum(r["dur"] for r in records)
    print(f"[{split}] resuming with {len(records)} clips, {total / 3600:.2f} h")
    
    dataset = load_dataset("ufal/parczech4speech-segmented", split=split, streaming=True)
    # Bypass torchcodec on Windows / torch 2.6.0
    dataset._info.features["flac"] = Value("binary")

    # Shuffles both the shard order and a rolling buffer of examples across shards
    shuffled_ds = dataset.shuffle(seed=seed, buffer_size=2000)

    samples = []
    speakers = []
    total_sec  = sum(r["dur"] for r in records)
    
    for ex in shuffled_ds:
        
        if total_sec >= target_sec:
            break
        
        meta = ex["json"]
        
        # filter
        if keep(meta, min_dur, max_dur) == False:
            continue

        speaker = meta["speaker"]
        speakers.append(speaker)
        
        url = ex.get("__url__")
        key = ex.get("__key__")
        
        if url is None or key is None:
            raise KeyError("example lacks __url__/__key__; cannot build a unique id")
        
        shard = Path(str(url)).stem
        
        audio_rel_path = f"{split}/{shard}/{key}.flac"
        id = f"{shard}_{key}"
        
        if id in done:
            continue
        
        done.add(id)
        
        record = save_sample(ex, split, audio_rel_path, id)
        
        records.append(record)
        samples.append(record)
        
        total_sec += meta["dur"]
        
    spk = Counter(r["speaker"] for r in records)
    top10 = sum(n for _, n in spk.most_common(10)) / max(len(records), 1)
    
    print(
        f"Sampled {len(samples)} utterances | "
        f"Total {len(records)} utterances"
        f"{total_sec / 3600.0:.2f}h ({total_sec:.1f}s) | "
        f"{len(set(speakers))} unique speakers"
        f"  top-10 speakers hold {top10:.0%} of clips"
    )

def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
      
def build_vocab(split, min_count):

    records = read_index(index_path(split))
    
    counts = Counter()
    
    texts = []
    for t in records:
        texts.append(t["norm_text"])
    
    for t in texts:
        counts.update(t)
        
    rare_tail = sorted(counts.items(), key=lambda x: x[1])[:30] # show 30 rarest characters 
    print("Rare character tail (char, count):", rare_tail)
    
    chars = []
    for ch, count in counts.items():
        if count >= min_count:
            chars.append(ch)
    chars.sort()

    vocab = ["<blank>"] + chars
    char_id = {}
    
    for idx, ch in enumerate(vocab):
        char_id[ch] = idx
        

    # Check that all 15 Czech diacritic letters are present
    missing = sorted(CZECH_DIACRITICS - set(char_id.keys()))
    if missing:
        print(f"WARNING: Missing Czech diacritic letters in vocab: {missing}")
    else:
        print("All 15 Czech diacritic letters are present in vocab.")

    VOCAB_PATH.write_text(json.dumps({"vocab": vocab}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print(f"saved {len(vocab)} symbols to {VOCAB_PATH}")
    
    return vocab

def keep(meta, min_dur, max_dur):

    dur = float(meta["dur"])
    if dur < min_dur or dur > max_dur : return False
            
    if meta["n_numbers"] > 0: return False
    
    text = meta["true_text"]
    norm_text = normalize(text)
    if norm_text == "": return False
    
    if any(c.isdigit() for c in text): 
        return False
        
    return True
        
def save_sample(sample, split, audio_rel_path, id):
    # TODO 13: make sure audio is mono 16 kHz (cast the audio column to a
    #          target sampling rate, or resample yourself).
    # TODO 14: save each clip (or its log-mel tensor) under out_dir.
    # TODO 15: write an index file: id, path, text, duration, speaker, split.
    meta = sample["json"]
    
    raw_flac = sample["flac"]
    audio_bytes = raw_flac["bytes"] if isinstance(raw_flac, dict) else raw_flac
    array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    
    
    if array.ndim > 1: # ensure mono
        array = array.mean(axis=1)
            
    if sr != TARGET_SR:    
        wav_tensor = torch.from_numpy(array)
        array = F.resample(wav_tensor, orig_freq=sr, new_freq=TARGET_SR).numpy()
        sr = TARGET_SR
        
    audio_dir = AUDIO_DIR / audio_rel_path
    
    audio_dir.parent.mkdir(parents=True, exist_ok=True)
           
    sf.write(audio_dir, array, TARGET_SR)
    
    metadata_path = index_path(split)
    
    record = {
        "id": id,
        "split": split,
        "dur": meta["dur"],
        "text": meta["true_text"],
        "speaker": meta["speaker"],
        "norm_text": normalize(meta["true_text"]),
        "seg_edit_dist": meta["seg_edit_dist"],
        "path" : audio_rel_path,
    }
    with open(metadata_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        
    return record

def check():
    train_path = index_path("train")
    val_path = index_path("validation")
    train = read_index(train_path)
    val = read_index(val_path)

    ### checking if there are speakers in both train and val sets
    train_speakers = set()
    for r in train:
        train_speakers.add(r["speaker"])
    val_speakers = set()
    for r in val:
        val_speakers.add(r["speaker"])

    overlap = train_speakers & val_speakers
    sorted_overlap = sorted(overlap)
    first_ten_overlap = sorted_overlap[:10]
    print(f"speaker overlap train/validation: {len(overlap)} {first_ten_overlap}")

    cps_list = []
    for r in train:
        rate = len(r["norm_text"]) / r["dur"]
        cps_list.append(rate)
    cps = sorted(cps_list)

    def pick(q):
        raw_index = int(q * len(cps))
        last_index = len(cps) - 1
        clamped_index = min(raw_index, last_index)
        return cps[clamped_index]

    p50 = pick(0.5)
    p90 = pick(0.9)
    p99 = pick(0.99)
    max_cps = cps[-1]
    print(
        f"chars/sec  p50 {p50:.1f}  p90 {p90:.1f}  "
        f"p99 {p99:.1f}  max {max_cps:.1f}   (output frames/s: 25 at 4x)"
    )

    featurizer = LogMelFeaturizer()
    ds = CzechSpeechDS("train", featurizer)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_function)
    loader_iter = iter(loader)
    batch = next(loader_iter)

    feats, flens, targets, tlens = batch

    feats_shape = tuple(feats.shape)
    flens_list = flens.tolist()
    print("features", feats_shape, "| feat_lens", flens_list)

    targets_shape = tuple(targets.shape)
    tlens_list = tlens.tolist()
    sum_matches = int(tlens.sum()) == targets.numel()
    print("targets", targets_shape, "| target_lens", tlens_list, "| sum ok:", sum_matches)

    first_target_len = tlens[0]
    first_target_ids = targets[:first_target_len].tolist()
    decoded_text = ds.decode(first_target_ids)
    expected_text = ds.samples[0]["norm_text"]

    print("first transcript:", decoded_text)
    print("vs index        :", expected_text)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--split", default="train")
    c.add_argument("--hours", type=float, default=1.0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--min-dur", type=float, default=1.0)
    c.add_argument("--max-dur", type=float, default=20.0)
    c.add_argument("--max-seg-edit", type=int, default=0)
    v = sub.add_parser("vocab")
    v.add_argument("--min-count", type=int, default=1)
    sub.add_parser("check")
    a = ap.parse_args()

    if a.cmd == "cache":
        sample_subset(a.hours, a.split, a.seed, a.min_dur, a.max_dur,)
    elif a.cmd == "vocab":
        build_vocab("train", a.min_count)
    elif a.cmd == "check":
        check()







