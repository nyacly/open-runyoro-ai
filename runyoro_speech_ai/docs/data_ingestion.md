# Data Ingestion and Preprocessing Pipeline

## Overview

This pipeline is designed to gather audio data from various sources (YouTube, local files), process it into a standardized format suitable for Automatic Speech Recognition (ASR) model training, and prepare a manifest file listing the processed audio data.

The key stages are:
1.  **Data Acquisition:**
    *   Downloads audio from YouTube videos.
    *   Ingests audio/video files from a local directory.
2.  **Audio Conversion:** Converts all acquired media into 16kHz mono WAV files. This is a common standard for ASR tasks.
3.  **Audio Segmentation:** Splits the converted WAV files into smaller segments based on silence detection and duration constraints. This helps create manageable audio clips for training.
4.  **Manifest Generation:** Creates a JSONL (JSON Lines) manifest file, where each line contains the absolute path to a processed audio segment and its duration in seconds.

## Prerequisites

*   **Python:** Python 3.8+ is recommended.
*   **System Dependencies:**
    *   `ffmpeg`: This is a crucial system dependency. `pydub`, used for audio processing, relies heavily on `ffmpeg` for handling various audio and video formats and for conversion tasks. Ensure `ffmpeg` is installed and accessible in your system's PATH.
*   **Python Packages:** All required Python packages are listed in `requirements.txt` in the project root. Install them using:
    ```bash
    pip install -r requirements.txt 
    ```
    (Note: The creation of `requirements.txt` faced issues in the initial setup due to environment constraints, but a functional environment would have this.)

## Directory Structure

The pipeline uses the following key directories within `runyoro_speech_ai/data/`:

*   `raw/youtube_downloads/`: Default location for audio extracted from YouTube videos.
*   `raw/local_uploads_sample/`: A sample directory where users can place their local audio/video files for ingestion.
*   `raw/local_ingested/`: Default location where local files are copied after initial ingestion.
*   `processed/converted_audio_16khz_mono/`: Default location for WAV files after conversion to 16kHz mono.
*   `processed/segmented_audio/`: Default location for segmented audio clips.
*   `manifest/`: Default location for the output manifest file (e.g., `audio_manifest.jsonl`).

These paths are defaults and can be overridden via command-line arguments in the main script.

## Main Orchestration Script (`main_ingest.py`)

The primary script to run the entire data ingestion and preprocessing pipeline is `runyoro_speech_ai/data_ingestion/main_ingest.py`.

### Command-Line Interface (CLI)

The script offers a flexible command-line interface to control various aspects of the pipeline. Below is the help output detailing all available arguments:

```
usage: main_ingest.py [-h] [--skip-youtube-download] [--skip-local-ingest]
                      [--skip-conversion] [--skip-segmentation]
                      [--skip-manifest] [--yt-urls URL [URL ...]]
                      [--yt-url-file FILE] [--yt-output-dir YT_OUTPUT_DIR]
                      [--local-upload-dir LOCAL_UPLOAD_DIR]
                      [--local-ingest-target-dir LOCAL_INGEST_TARGET_DIR]
                      [--conversion-input-dir CONVERSION_INPUT_DIR]
                      [--conversion-output-dir CONVERSION_OUTPUT_DIR]
                      [--segmentation-input-dir SEGMENTATION_INPUT_DIR]
                      [--segmentation-output-dir SEGMENTATION_OUTPUT_DIR]
                      [--min-silence-len MIN_SILENCE_LEN]
                      [--silence-thresh SILENCE_THRESH]
                      [--keep-silence KEEP_SILENCE]
                      [--min-segment-duration MIN_SEGMENT_DURATION]
                      [--max-segment-duration MAX_SEGMENT_DURATION]
                      [--target-split-duration TARGET_SPLIT_DURATION]
                      [--manifest-input-dir MANIFEST_INPUT_DIR]
                      [--manifest-output-file MANIFEST_OUTPUT_FILE]

Runyoro Speech AI Data Ingestion Pipeline.

options:
  -h, --help            show this help message and exit
  --skip-youtube-download
                        Skip YouTube video audio download stage.
  --skip-local-ingest   Skip local media file ingestion stage.
  --skip-conversion     Skip audio conversion to 16kHz mono WAV.
  --skip-segmentation   Skip audio segmentation stage.
  --skip-manifest       Skip manifest generation stage.

YouTube Download Options:
  --yt-urls URL [URL ...]
                        List of YouTube video URLs.
  --yt-url-file FILE    Path to a file containing YouTube URLs (one per line).
  --yt-output-dir YT_OUTPUT_DIR
                        Directory to save downloaded YouTube audio. Default:
                        /app/runyoro_speech_ai/data/raw/youtube_downloads

Local File Ingestion Options:
  --local-upload-dir LOCAL_UPLOAD_DIR
                        Directory containing user-uploaded local media files.
                        Default:
                        /app/runyoro_speech_ai/data/raw/local_uploads_sample
  --local-ingest-target-dir LOCAL_INGEST_TARGET_DIR
                        Directory to copy raw local media files to. Default:
                        /app/runyoro_speech_ai/data/raw/local_ingested

Audio Conversion Options (to 16kHz mono WAV):
  --conversion-input-dir CONVERSION_INPUT_DIR
                        Directory of raw media for conversion. If not set,
                        uses outputs from YouTube & Local Ingest stages if
                        they ran.
  --conversion-output-dir CONVERSION_OUTPUT_DIR
                        Directory to save converted 16kHz mono WAV files.
                        Default: /app/runyoro_speech_ai/data/processed/convert
                        ed_audio_16khz_mono

Audio Segmentation Options:
  --segmentation-input-dir SEGMENTATION_INPUT_DIR
                        Directory of 16kHz mono WAVs for segmentation. If not
                        set, uses --conversion-output-dir. Default: (output of
                        conversion stage)
  --segmentation-output-dir SEGMENTATION_OUTPUT_DIR
                        Directory to save segmented audio clips. Default:
                        /app/runyoro_speech_ai/data/processed/segmented_audio
  --min-silence-len MIN_SILENCE_LEN
                        Min silence length (ms) for VAD. Default: 700.
  --silence-thresh SILENCE_THRESH
                        Silence threshold (dBFS) for VAD. Default: -45.
  --keep-silence KEEP_SILENCE
                        Keep silence (ms) at segment ends. Default: 250.
  --min-segment-duration MIN_SEGMENT_DURATION
                        Min segment duration (ms). Default: 2000 (2s).
  --max-segment-duration MAX_SEGMENT_DURATION
                        Max segment duration (ms) before force split. Default:
                        20000 (20s).
  --target-split-duration TARGET_SPLIT_DURATION
                        Target duration (ms) for force-split segments.
                        Default: 10000 (10s).

Manifest Generation Options:
  --manifest-input-dir MANIFEST_INPUT_DIR
                        Directory of final segmented audio clips for manifest.
                        If not set, uses --segmentation-output-dir. Default:
                        (output of segmentation stage)
  --manifest-output-file MANIFEST_OUTPUT_FILE
                        Path to save the generated manifest JSONL file.
                        Default: /app/runyoro_speech_ai/data/manifest/audio_ma
                        nifest.jsonl
```

### Example Usage

1.  **Run the full pipeline with default paths, providing YouTube URLs:**
    ```bash
    python runyoro_speech_ai/data_ingestion/main_ingest.py \
        --yt-urls "https://www.youtube.com/watch?v=VIDEO_ID_1" "https://www.youtube.com/watch?v=VIDEO_ID_2" \
        --local-upload-dir path/to/your/local_files/
    ```
    *(Ensure `path/to/your/local_files/` contains some media for the local ingestion part, or skip it.)*

2.  **Run only specific stages (e.g., conversion and segmentation), specifying input/output:**
    ```bash
    python runyoro_speech_ai/data_ingestion/main_ingest.py \
        --skip-youtube-download \
        --skip-local-ingest \
        --conversion-input-dir path/to/raw_audio_for_conversion/ \
        --conversion-output-dir path/to/converted_audio/ \
        --segmentation-output-dir path/to/segmented_clips/ \
        --skip-manifest 
    ```

3.  **Adjust segmentation parameters:**
    ```bash
    python runyoro_speech_ai/data_ingestion/main_ingest.py \
        --skip-youtube-download --skip-local-ingest --skip-conversion --skip-manifest \
        --segmentation-input-dir path/to/converted_audio/ \
        --segmentation-output-dir path/to/segmented_clips/ \
        --min-silence-len 600 \
        --silence-thresh -50 \
        --min-segment-duration 1500 \
        --max-segment-duration 15000 \
        --target-split-duration 8000
    ```

## Individual Scripts

The `main_ingest.py` script orchestrates the following individual scripts, which can also be run standalone for specific tasks (though paths and parameters would need to be managed carefully):

*   **`runyoro_speech_ai/data_ingestion/download_youtube.py`**: Downloads audio from YouTube.
*   **`runyoro_speech_ai/data_ingestion/process_local_files.py`**: Copies media files from a user-specified local directory.
*   **`runyoro_speech_ai/data_ingestion/preprocess_audio.py`**: Handles audio conversion to 16kHz mono WAV and subsequent segmentation.
*   **`runyoro_speech_ai/scripts/generate_manifest.py`**: Creates the final JSONL manifest file from the segmented audio clips.

## Output

The final output of the pipeline is a **manifest file** in JSONL format (typically saved to `runyoro_speech_ai/data/manifest/audio_manifest.jsonl` or as specified). Each line in this file is a JSON object representing one audio segment, with the following structure:

```json
{"audio_filepath": "/path/to/project/runyoro_speech_ai/data/processed/segmented_audio/segment_filename.wav", "duration": 10.532}
```

*   `audio_filepath`: The absolute path to the processed audio segment (WAV file).
*   `duration`: The duration of the audio segment in seconds.

This manifest file is essential for training ASR models, as it provides the model with the locations and lengths of the audio data.

## Troubleshooting/Common Issues

*   **`ffmpeg` Not Found/Errors:**
    *   Many audio processing steps rely on `ffmpeg`. If it's not installed or not in your system's PATH, `pydub` will fail, leading to errors during conversion or even when trying to read some source formats.
    *   **Solution:** Install `ffmpeg` (and `ffprobe`) from your system's package manager (e.g., `sudo apt-get install ffmpeg`) or download static builds and ensure they are in your PATH.
*   **YouTube Download Failures:**
    *   Videos might be unavailable, private, or region-restricted. `yt-dlp` usually provides informative error messages.
    *   Network issues can interrupt downloads.
    *   **Solution:** Check the URL accessibility in a browser. Ensure your network connection is stable. `yt-dlp` might need updates (`pip install --upgrade yt-dlp`) to handle changes in YouTube's platform.
*   **Permissions:** Ensure the script has read permissions for input directories and write permissions for output directories.
*   **File Paths:** When running individual scripts or the main orchestrator, always ensure that relative paths are correctly resolved or use absolute paths for clarity, especially for input/output directories. The default paths in `main_ingest.py` are constructed relative to the project structure.The documentation file `runyoro_speech_ai/docs/data_ingestion.md` has been successfully created with the detailed content including the `--help` output from `main_ingest.py`.

**Phase 2: Update Main `README.md`**

Now, I'll add a link to this new documentation file in the main `runyoro_speech_ai/README.md`. I'll need to read its current content first.
