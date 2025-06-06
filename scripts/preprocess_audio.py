#!/usr/bin/env python
"""Simple audio preprocessing example.

This script converts an input audio file to a 16 kHz mono WAV file.

Usage:
    python scripts/preprocess_audio.py input_audio output.wav
"""

import sys
from pydub import AudioSegment


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Load ``input_path`` and save it as 16 kHz mono WAV to ``output_path``."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_audio.py input_audio output.wav")
        sys.exit(1)
    convert_to_wav(sys.argv[1], sys.argv[2])
