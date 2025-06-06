import json
import os
import json
import sys
from unittest import mock

import pytest
from pydub import generators
import yaml

from runyoro_speech_ai.asr_finetune import build_manifest
from runyoro_speech_ai.asr_finetune import train_ctc


@pytest.fixture
def sample_bible_audio_text(tmp_path):
    audio_base = tmp_path / "audio" / "FSID"
    text_base = tmp_path / "text" / "TXTID"
    audio_base.mkdir(parents=True)
    text_base.mkdir(parents=True)

    audio_path = audio_base / "FSID_MAT_001.wav"
    tone = generators.Sine(440).to_audio_segment(duration=1000).set_channels(1).set_frame_rate(16000)
    tone.export(audio_path, format="wav")

    text_path = text_base / "MAT_001.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("1 In the beginning.\n2 Another verse!")

    return audio_base.parent, text_base.parent, audio_path, text_path


def test_build_manifest(sample_bible_audio_text, tmp_path):
    audio_dir_base, text_dir_base, audio_file, text_file = sample_bible_audio_text
    manifest = tmp_path / "manifest.json"

    count = build_manifest.build_manifest(audio_dir_base, text_dir_base, "FSID", "TXTID", manifest)
    assert count == 1
    data = [json.loads(manifest.read_text().splitlines()[0])]
    assert data[0]["id"] == "FSID_MAT_001"
    assert os.path.abspath(audio_file) == data[0]["wav"]
    assert data[0]["wrd"] == "in the beginning. another verse!"


def test_train_ctc_mock(tmp_path):
    # Prepare tiny manifest and YAML
    audio_dir = tmp_path / "audio" / "FSID"
    text_dir = tmp_path / "text" / "TXTID"
    audio_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    tone = generators.Sine(440).to_audio_segment(duration=500).set_channels(1).set_frame_rate(16000)
    wav_path = audio_dir / "FSID_MAT_001.wav"
    tone.export(wav_path, format="wav")
    txt_path = text_dir / "MAT_001.txt"
    txt_path.write_text("1 sample text")
    manifest = tmp_path / "manifest.json"
    build_manifest.build_manifest(audio_dir.parent, text_dir.parent, "FSID", "TXTID", manifest)

    hparams = {
        "data_folder": str(tmp_path),
        "train_manifest": "manifest.json",
        "tokenizer_model_dir": str(tmp_path / "tok"),
        "tokenizer_model_prefix": "spm_test",
        "tokenizer_vocab_size": 12,
        "ssl_model_hub": "hf-internal-testing/tiny-random-wav2vec2",
        "output_folder": str(tmp_path / "out"),
    }
    hparams_file = tmp_path / "h.yaml"
    hparams_file.write_text(yaml.safe_dump(hparams))

    dummy_sb = mock.MagicMock()
    patches = [mock.patch.dict(sys.modules, {"speechbrain": dummy_sb, "torch": dummy_sb})]
    for p in patches:
        p.start()

    try:
        sys.argv = ["train_ctc.py", str(hparams_file)]
        train_ctc.main()
    finally:
        for p in patches:
            p.stop()

