import pytest
import os
import sys
import json
import subprocess
import shutil
from unittest import mock
from pydub import AudioSegment, generators

# Ensure the scripts and data_ingestion directories are in the Python path for imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
data_ingestion_dir = os.path.join(project_root, 'data_ingestion')
scripts_dir = os.path.join(project_root, 'scripts')

if data_ingestion_dir not in sys.path:
    sys.path.insert(0, data_ingestion_dir)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import functions to be tested
from preprocess_audio import run_conversion_stage, run_segmentation_stage, TARGET_SAMPLING_RATE
from generate_manifest import create_audio_manifest
from download_youtube import orchestrate_youtube_downloads
from process_local_files import ingest_local_media, SUPPORTED_EXTENSIONS as LOCAL_SUPPORTED_EXTENSIONS


# --- Fixtures ---

@pytest.fixture(scope="module")
def module_tmp_path(tmp_path_factory):
    """A temporary path for the entire test module, cleaned up at the end."""
    return tmp_path_factory.mktemp("data_ingestion_tests_module")

@pytest.fixture(scope="module")
def audio_fixtures_dir(module_tmp_path):
    """Creates a subdirectory within the module's tmp_path for generated audio fixtures."""
    fixtures_path = module_tmp_path / "audio_fixtures"
    fixtures_path.mkdir(parents=True, exist_ok=True)
    return fixtures_path

@pytest.fixture(scope="module")
def sample_audio_mp3(audio_fixtures_dir):
    """Generate a short MP3 file for testing conversion."""
    sine_wave = generators.Sine(440).to_audio_segment(duration=1000).set_channels(2).set_frame_rate(44100)
    filepath = audio_fixtures_dir / "sample_stereo_44100hz.mp3"
    try:
        sine_wave.export(filepath, format="mp3")
    except Exception as e:
        pytest.skip(f"Skipping MP3 generation due to pydub/ffmpeg issue: {e}") # Skip if ffmpeg is missing
    return str(filepath)

@pytest.fixture(scope="module")
def sample_audio_wav_for_segmentation(audio_fixtures_dir):
    """Generate a WAV file with silence for testing segmentation."""
    tone1 = generators.Sine(440).to_audio_segment(duration=500).set_channels(1).set_frame_rate(TARGET_SAMPLING_RATE)
    silence = AudioSegment.silent(duration=800, frame_rate=TARGET_SAMPLING_RATE) # min_silence_len is 700ms
    tone2 = generators.Sine(330).to_audio_segment(duration=500).set_channels(1).set_frame_rate(TARGET_SAMPLING_RATE)
    combined = tone1 + silence + tone2
    filepath = audio_fixtures_dir / "sample_for_segmentation.wav"
    combined.export(filepath, format="wav")
    return str(filepath)

@pytest.fixture(scope="module")
def sample_audio_wav_no_silence(audio_fixtures_dir, tmp_path_factory):
    """Generate a WAV file with no significant silence for testing segmentation edge case.
       This fixture is function-scoped if it needs a unique directory.
    """
    # Create a unique subdir for this specific test's input file to avoid cross-test contamination
    # This fixture will return the path to the file *inside* its own unique directory.
    specific_input_dir = tmp_path_factory.mktemp("no_silence_input")
    tone = generators.Sine(440).to_audio_segment(duration=2000).set_channels(1).set_frame_rate(TARGET_SAMPLING_RATE) # 2 seconds long
    filepath = specific_input_dir / "sample_no_silence.wav"
    tone.export(filepath, format="wav")
    return str(filepath)


@pytest.fixture(scope="module")
def sample_wav_for_manifest(audio_fixtures_dir):
    duration_ms = 1500
    tone = generators.Square(220).to_audio_segment(duration=duration_ms).set_channels(1).set_frame_rate(TARGET_SAMPLING_RATE)
    filepath = audio_fixtures_dir / "manifest_sample_1.wav"
    tone.export(filepath, format="wav")
    return str(filepath), duration_ms / 1000.0

@pytest.fixture(scope="module")
def sample_wav_for_manifest_2(audio_fixtures_dir):
    duration_ms = 2500
    tone = generators.Sawtooth(330).to_audio_segment(duration=duration_ms).set_channels(1).set_frame_rate(TARGET_SAMPLING_RATE)
    filepath = audio_fixtures_dir / "manifest_sample_2.wav"
    tone.export(filepath, format="wav")
    return str(filepath), duration_ms / 1000.0

@pytest.fixture(scope="function")
def local_files_fixture_dir(tmp_path):
    source_dir = tmp_path / "local_uploads_source"
    source_dir.mkdir()
    try:
        (generators.Sine(440).to_audio_segment(duration=100)
         .export(source_dir / "audio1.mp3", format="mp3"))
        (generators.Sine(440).to_audio_segment(duration=100)
         .export(source_dir / "audio2.wav", format="wav"))
        (generators.Sine(440).to_audio_segment(duration=100) 
         .export(source_dir / "video1.mp4", format="mp4")) 
    except Exception as e:
         pytest.skip(f"Skipping local_files_fixture_dir generation due to pydub/ffmpeg issue: {e}")

    with open(source_dir / "document.txt", "w") as f: f.write("Test document.")
    with open(source_dir / "script.py", "w") as f: f.write("print('hello')")
    return source_dir

# --- Tests for process_local_files.py ---
def test_local_file_ingestion(local_files_fixture_dir, tmp_path):
    user_upload_dir = local_files_fixture_dir
    raw_files_target_dir = tmp_path / "local_ingest_target"
    ingest_local_media(user_upload_dir=str(user_upload_dir), raw_files_target_dir=str(raw_files_target_dir))
    assert raw_files_target_dir.exists()
    copied_files = os.listdir(raw_files_target_dir)
    assert len(copied_files) == 3
    expected_copied = ["audio1.mp3", "audio2.wav", "video1.mp4"]
    for fname in expected_copied: assert fname in copied_files
    assert "document.txt" not in copied_files and "script.py" not in copied_files

# --- Tests for preprocess_audio.py ---

def test_audio_conversion(sample_audio_mp3, tmp_path):
    if not os.path.exists(sample_audio_mp3): # Handle skip from fixture
        pytest.skip("Skipping test_audio_conversion as MP3 fixture generation failed (likely no ffmpeg).")
    input_dir = os.path.dirname(sample_audio_mp3)
    conversion_output_dir = tmp_path / "converted_audio"
    run_conversion_stage(input_dir=str(input_dir), conversion_output_dir=str(conversion_output_dir))
    original_basename = os.path.splitext(os.path.basename(sample_audio_mp3))[0]
    expected_output_filename = f"{original_basename}.wav"
    output_filepath = conversion_output_dir / expected_output_filename
    assert output_filepath.exists()
    converted_audio = AudioSegment.from_file(output_filepath)
    assert converted_audio.frame_rate == TARGET_SAMPLING_RATE
    assert converted_audio.channels == 1
    assert output_filepath.suffix.lower() == ".wav"

def test_audio_segmentation_basic(sample_audio_wav_for_segmentation, tmp_path):
    # Use a specific input directory for this test to avoid contamination
    test_specific_input_dir = tmp_path / "seg_basic_input"
    test_specific_input_dir.mkdir()
    shutil.copy(sample_audio_wav_for_segmentation, test_specific_input_dir)

    segmentation_output_dir = tmp_path / "segmented_audio_basic"
    min_silence_len, silence_thresh, keep_silence = 700, -45, 250
    min_duration_ms, max_duration_ms, target_split_duration_ms = 300, 15000, 10000
    run_segmentation_stage(
        segmentation_input_dir=str(test_specific_input_dir), segmentation_output_dir=str(segmentation_output_dir),
        min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=keep_silence,
        min_duration_ms=min_duration_ms, max_duration_ms=max_duration_ms, target_split_duration_ms=target_split_duration_ms
    )
    output_files = list(segmentation_output_dir.glob("*.wav"))
    assert len(output_files) == 2, f"Expected 2 segments, found {len(output_files)}"
    for seg_file in output_files:
        segment_audio = AudioSegment.from_file(seg_file)
        assert (500 - 50) <= len(segment_audio) <= (500 + 2 * keep_silence + 50)

def test_audio_segmentation_no_silence_force_split(sample_audio_wav_no_silence, tmp_path):
    # sample_audio_wav_no_silence is already in its own unique directory thanks to the fixture modification
    input_dir = os.path.dirname(sample_audio_wav_no_silence) 
    segmentation_output_dir = tmp_path / "segmented_audio_no_silence"

    min_silence_len, silence_thresh, keep_silence = 700, -45, 250
    min_duration_ms = 500 
    max_duration_ms = 1500 
    target_split_duration_ms = 800

    run_segmentation_stage(
        segmentation_input_dir=str(input_dir), segmentation_output_dir=str(segmentation_output_dir),
        min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=keep_silence,
        min_duration_ms=min_duration_ms, max_duration_ms=max_duration_ms, target_split_duration_ms=target_split_duration_ms
    )
    output_files = list(segmentation_output_dir.glob("*.wav"))
    assert len(output_files) == 2, f"Expected 2 segments after force splitting, found {len(output_files)}"
    for seg_file in output_files:
        segment_audio = AudioSegment.from_file(seg_file)
        assert len(segment_audio) == target_split_duration_ms

# --- Tests for generate_manifest.py ---

def test_manifest_generation(sample_wav_for_manifest, sample_wav_for_manifest_2, tmp_path):
    segmented_audio_dir = tmp_path / "segmented_for_manifest"
    segmented_audio_dir.mkdir()
    fixture1_path, fixture1_duration = sample_wav_for_manifest
    fixture2_path, fixture2_duration = sample_wav_for_manifest_2
    shutil.copy(fixture1_path, segmented_audio_dir / os.path.basename(fixture1_path))
    shutil.copy(fixture2_path, segmented_audio_dir / os.path.basename(fixture2_path))
    manifest_output_file = tmp_path / "test_manifest.jsonl"
    create_audio_manifest(input_dir=str(segmented_audio_dir), output_file_path=str(manifest_output_file))
    assert manifest_output_file.exists()
    processed_files_in_manifest = {}
    line_count = 0
    with open(manifest_output_file, 'r') as f:
        for line in f:
            line_count +=1
            entry = json.loads(line)
            assert "audio_filepath" in entry and "duration" in entry
            processed_files_in_manifest[os.path.basename(entry["audio_filepath"])] = entry["duration"]
    assert line_count == 2
    assert os.path.basename(fixture1_path) in processed_files_in_manifest
    assert abs(processed_files_in_manifest[os.path.basename(fixture1_path)] - fixture1_duration) < 0.01
    assert os.path.basename(fixture2_path) in processed_files_in_manifest
    assert abs(processed_files_in_manifest[os.path.basename(fixture2_path)] - fixture2_duration) < 0.01

# --- Tests for download_youtube.py (Mocking) ---

@mock.patch('download_youtube.subprocess.run')
def test_youtube_download_call_single_url(mock_subprocess_run, tmp_path):
    sample_url = "https://www.youtube.com/watch?v=test1234"
    output_dir = tmp_path / "youtube_downloads"
    mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="Download successful", stderr="")
    orchestrate_youtube_downloads(urls_list=[sample_url], output_dir=str(output_dir))
    assert mock_subprocess_run.called
    called_args, _ = mock_subprocess_run.call_args
    command_list = called_args[0]
    assert "yt-dlp" in command_list and "-x" in command_list and sample_url in command_list
    try:
        output_flag_index = command_list.index('-o')
        output_template_arg = command_list[output_flag_index + 1]
        assert str(output_dir) in output_template_arg and "%(id)s.%(ext)s" in output_template_arg
    except ValueError: pytest.fail("'-o' flag not found or path incorrect.")

@mock.patch('download_youtube.subprocess.run')
def test_youtube_download_call_from_file(mock_subprocess_run, tmp_path):
    urls = ["https://www.youtube.com/watch?v=vid1", "https://www.youtube.com/watch?v=vid2"]
    url_file = tmp_path / "urls.txt"
    with open(url_file, "w") as f:
        for url in urls: f.write(url + "\n")
    output_dir = tmp_path / "youtube_downloads_from_file"
    mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="Downloaded", stderr="")
    orchestrate_youtube_downloads(url_file_path=str(url_file), output_dir=str(output_dir))
    assert mock_subprocess_run.call_count == len(urls)
    for i, url in enumerate(urls):
        called_args, _ = mock_subprocess_run.call_args_list[i]
        command_list = called_args[0]
        assert url in command_list

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
