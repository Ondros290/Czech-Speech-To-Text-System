import torch
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_DIR / "vendor" / "WhisperRT-Streaming"

sys.path.insert(0, str(VENDOR_DIR))

import whisper_rt

class StreamingASR:
    #def __init__(self):
        
    
    def start(self):
        """Initialize the model and reset streaming state."""
        model_size = "small" # model size
        chunk_size = 300 # chunk size in milliseconds
        multilingual = False # currently on large-v2_300msec supports other languages than english.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper_rt.load_streaming_model(name=model_size,
                                        gran=chunk_size,
                                        multilingual=multilingual,
                                        device=device)
        print("LOADED successfully")

    def accept_audio(self, samples, audio_end_time: float):
        """Consume new audio and return zero or more hypotheses."""

    def finalize(self):
        """Flush remaining audio and return the final hypothesis."""
        
strm = StreamingASR()
strm.start()