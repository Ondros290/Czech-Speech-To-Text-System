class StreamingASR:
    def start(self, sample_rate: int, language: str):
        """Initialize the model and reset streaming state."""

    def accept_audio(self, samples, audio_end_time: float):
        """Consume new audio and return zero or more hypotheses."""

    def finalize(self):
        """Flush remaining audio and return the final hypothesis."""