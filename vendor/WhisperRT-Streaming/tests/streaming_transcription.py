import os
import sys
sys.path.append("./")

import torch
import whisper_rt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper_rt.load_streaming_model("small", 300, False, False, device)
texts = model.transcribe(simulate_stream=True, wav_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "jfk.wav"), beam_size=3, ca_kv_cache=True)
print(texts)