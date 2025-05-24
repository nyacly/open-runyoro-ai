# Runyoro Speech AI Project

This project aims to develop robust Artificial Intelligence capabilities for the Runyoro language, starting with Automatic Speech Recognition (ASR) and laying the groundwork for future speech generation (Text-to-Speech) and natural language understanding (NLU).

## Project Phases (High-Level)

1.  **Data Ingestion & Preprocessing:** Automated pipeline for acquiring and preparing Runyoro audio data.
2.  **Self-Supervised Learning (SSL) Model Training:** Pre-training a model on unlabeled Runyoro audio.
3.  **ASR Fine-tuning:** Fine-tuning the SSL model for speech recognition.
4.  **Inference System:** CLI/API for transcribing Runyoro audio.
5.  **Documentation & Scalability:** Comprehensive docs and cloud scaling plan.

## Setup

(Instructions to be added once environment setup is stable and `requirements.txt` is finalized.)

## Data Ingestion Pipeline

A detailed guide on the data ingestion and preprocessing pipeline, including setup, usage of the main orchestration script (`data_ingestion/main_ingest.py`), command-line arguments, and troubleshooting, can be found in:

*   **[Data Ingestion and Preprocessing Documentation](./docs/data_ingestion.md)**

This pipeline handles:
*   Downloading audio from YouTube.
*   Ingesting local audio/video files.
*   Standardizing audio to 16kHz mono WAV format.
*   Segmenting audio based on silence and duration.
*   Generating a manifest file for ASR model training.

## Project Directory Structure

This project follows a structured layout to keep code, data, and documentation organized:

-   **`runyoro_speech_ai/`**: The root directory of the project.
    -   **`data/`**: Contains all data related to the project.
        -   `data/raw/`: Raw data as initially acquired.
            -   `data/raw/youtube_downloads/`: (Default) Stores audio files downloaded from YouTube.
            -   `data/raw/local_ingested/`: (Default) For user-provided local audio/video files after initial copying.
            -   `data/raw/local_uploads_sample/`: A sample directory illustrating where users might place files for local ingestion.
        -   `data/processed/`: Data that has undergone some form of processing.
            -   `data/processed/converted_audio_16khz_mono/`: (Default) Stores audio files converted to a consistent format (16kHz mono WAV).
            -   `data/processed/segmented_audio/`: (Default) Contains smaller audio segments derived from the processed audio, ready for manifest creation.
            -   `data/processed/ssl_dataset/`: Stores the Hugging Face `Dataset` specifically prepared for Self-Supervised Learning (SSL) model training.
            -   `data/processed/asr_dataset/`: (Future use) Will store datasets prepared for Automatic Speech Recognition (ASR) fine-tuning.
        -   `data/manifest/`: Holds manifest files (typically JSONL format) that list audio file paths and their metadata (like duration).
            -   `data/manifest/audio_manifest.jsonl`: (Default) Manifest generated from all segmented audio.
    -   **`data_ingestion/`**: Contains Python scripts responsible for data acquisition and initial preprocessing (downloading, format conversion, segmentation).
        -   `main_ingest.py`: The main orchestrator script for the data ingestion pipeline.
        -   `download_youtube.py`, `process_local_files.py`, `preprocess_audio.py`: Core component scripts for ingestion.
    -   **`ssl_training/`**: Scripts and documentation related to Self-Supervised Learning (SSL) model pre-training.
        -   `prepare_ssl_data.py`: Prepares the manifest data for SSL training.
        -   `train_ssl.py`: Script for running the SSL training.
        -   `README.md`: Documentation specific to SSL training.
    -   **`asr_finetuning/`**: (Future use) Will contain scripts and resources for fine-tuning models for Automatic Speech Recognition (ASR).
    -   **`inference/`**: (Future use) Will house scripts for using trained models to perform transcription.
    -   **`models/`**: Default directory for storing trained model checkpoints and related artifacts.
        -   `models/ssl/`: Specifically for SSL pre-trained models.
        *   `models/asr/`: (Future use) For ASR fine-tuned models.
    -   **`scripts/`**: Utility and helper scripts (e.g., `check_mps.py` for verifying Apple Silicon GPU support, `test_run_ssl.sh` for test runs, `generate_manifest.py`).
    -   **`docs/`**: Detailed documentation for specific parts of the pipeline (e.g., `data_ingestion.md`, setup guides).
    -   **`tests/`**: Contains unit tests and integration tests for the project's codebase.
        -   `tests/fixtures/`: (If used, though current tests generate fixtures on-the-fly) Sample data files for testing.
        -   `test_data_ingestion.py`: Example test file.
    -   **`.github/`**: (If using GitHub) Workflows for GitHub Actions (e.g., CI/CD).
    -   **`runyoro_env/` or `.venv/`**: (If following local setup guide) Python virtual environment directory (should be in `.gitignore`).
    -   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore (project-level and potentially a root-level one).
    -   **`README.md`**: The main landing page for the project, providing an overview and guidance (this file).
    -   **`requirements.txt`**: (Goal) Lists Python package dependencies (creation pending resolution of environment limitations).

## Quick Start: Training an SSL Model from YouTube Links

This guide provides a step-by-step walkthrough to download audio from YouTube, process it, and start Self-Supervised Learning (SSL) pre-training on your local machine (e.g., a MacBook Pro M4).

*(For more in-depth explanations of each stage, please refer to the documentation in the `docs/` and specific pipeline directories like `ssl_training/`.)*

### Step 1: Environment Setup

Before you begin, ensure your environment is correctly set up:

1.  **Clone the Repository:** If you haven't already, clone this project to your local machine.
2.  **Python Virtual Environment:**
    *   Navigate to the project root in your terminal.
    *   Create and activate a Python virtual environment:
        ```bash
        python3 -m venv runyoro_env
        source runyoro_env/bin/activate
        ```
3.  **Install Dependencies:**
    *   **`ffmpeg`**: Install `ffmpeg` (if not already present). On macOS with Homebrew: `brew install ffmpeg`
    *   **Python Packages**: Install required Python libraries. While a `requirements.txt` is the goal, you may need to install them manually based on the lists in `docs/data_ingestion.md` and `ssl_training/README.md`. Key packages include `torch`, `torchaudio`, `transformers`, `datasets`, `accelerate`, `yt-dlp`, `pydub`.
        *   Ensure your PyTorch installation supports MPS for Apple Silicon. See [PyTorch Get Started](https://pytorch.org/get-started/locally/).
4.  **Verify MPS (for Apple Silicon users):**
    ```bash
    python ./scripts/check_mps.py
    ```
    You should see confirmation that MPS is available and functional.

### Step 2: Prepare YouTube Links File

1.  Create a plain text file, for example, `youtube_links.txt`, inside the `runyoro_speech_ai/data_ingestion/` directory.
2.  Add one YouTube video URL per line in this file. These videos should contain the Runyoro speech you want to process.

    *Example `data_ingestion/youtube_links.txt`:*
    ```
    https://www.youtube.com/watch?v=xxxxxxxxx_01
    https://www.youtube.com/watch?v=xxxxxxxxx_02
    ```

### Step 3: Run Data Ingestion and Preprocessing

This step uses the `main_ingest.py` script to download audio from your list of YouTube links, convert it to the required audio format (16kHz mono WAV), segment it into smaller clips, and generate a manifest file.

Execute the following command from the project root directory:

```bash
python ./data_ingestion/main_ingest.py \
    --yt-url-file ./data_ingestion/youtube_links.txt \
    --skip-local-ingest  # Use this flag if you are only processing YouTube links
```

-   **Output:**
    -   Downloaded audio will be in `data/raw/youtube_downloads/`.
    -   Processed 16kHz mono WAVs in `data/processed/converted_audio_16khz_mono/`.
    -   Segmented audio clips in `data/processed/segmented_audio/`.
    -   A manifest file named `audio_manifest.jsonl` will be created in `data/manifest/`. This file lists all segmented audio clips and their durations, which is crucial for the next steps.

### Step 4: Prepare Dataset for SSL Training

This step takes the `audio_manifest.jsonl` and prepares the audio data specifically for the SSL model. This involves loading the audio segments and applying the feature extraction process defined by the pre-trained SSL model.

Execute the `prepare_ssl_data.py` script:

```bash
python ./ssl_training/prepare_ssl_data.py \
    --manifest_path ./data/manifest/audio_manifest.jsonl \
    --output_dir ./data/processed/ssl_dataset/ \
    --model_name_or_path facebook/wav2vec2-xls-r-300m \
    --max_duration_sec 15 \
    --min_duration_sec 2
```
*Note: For faster processing, you can add `--num_workers <number_of_cpu_cores>` (e.g., `--num_workers 4`) to the command above.*

-   **Output:** This will create a Hugging Face `Dataset` formatted for training, saved in the `./data/processed/ssl_dataset/` directory.

### Step 5: Start SSL Model Pre-training

Now you can begin the actual SSL pre-training. This process takes the base `facebook/wav2vec2-xls-r-300m` model and continues its training using your prepared Runyoro audio data. This helps the model adapt to the specific acoustic characteristics of Runyoro.

Execute the `train_ssl.py` script:

```bash
python ./ssl_training/train_ssl.py \
    --processed_dataset_path ./data/processed/ssl_dataset/ \
    --model_name_or_path facebook/wav2vec2-xls-r-300m \
    --output_dir ./models/ssl/my_runyoro_ssl_model/ \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --fp16 \
    --seed 42
```
*Note: For faster data loading during training, you can add `--dataloader_num_workers <number_of_cpu_cores>` (e.g., `--dataloader_num_workers 4`) to the command above.*

-   **Important for MacBook M4 users:**
    -   Start with a small `--per_device_train_batch_size` (e.g., 1 or 2).
    -   Use `--gradient_accumulation_steps` to achieve a larger effective batch size.
    -   `--fp16` is recommended for mixed-precision training on MPS.
    -   Monitor your system resources (memory, CPU/GPU via Activity Monitor). If you encounter "out of memory" errors, reduce the batch size or gradient accumulation steps.

### Step 6: Monitoring Training & Outputs

-   **Console Logs:** Training progress, including loss and learning rate, will be printed to your terminal.
-   **Model Checkpoints:** Checkpoints will be saved periodically in subdirectories within your specified `--output_dir` (e.g., `./models/ssl/my_runyoro_ssl_model/checkpoint-500/`).
-   **Final Model:** The final trained SSL model will be saved in the root of the `--output_dir` once training is complete. This model can then be used as a base for ASR fine-tuning (Phase 3).

### Managing YouTube Links and Re-processing

When working with lists of YouTube links for data collection, consider the following:

-   **No Automatic Tracking of Processed URLs:** The current scripts do **not** automatically track which individual YouTube URLs have been successfully processed and included in a specific training run.
-   **Re-running Ingestion:**
    -   If you provide the same `youtube_links.txt` file to `main_ingest.py` multiple times, `yt-dlp` (the underlying YouTube downloader) is generally smart enough to avoid re-downloading the full video/audio if the file already exists in `data/raw/youtube_downloads/` with the same name (e.g., `videoid.opus`). However, it might still check the URL.
    -   Regardless of download, the subsequent audio processing and segmentation steps *will* run on all audio files found based on the provided links, meaning existing audio will be re-segmented and re-included in the generated `audio_manifest.jsonl`.
-   **Avoiding Re-processing or Using New Links:**
    -   If you wish to avoid re-processing certain URLs or want to ensure only new links are used for a particular data preparation run, you should **manually manage your `youtube_links.txt` file.**
    -   **Strategies:**
        -   **Remove/Comment Out:** Delete or comment out (e.g., with a `#` at the beginning of the line, though the script doesn't explicitly support comments) URLs from `youtube_links.txt` that you consider "processed" or do not want to include in a new run.
        -   **Separate Files:** Maintain different list files for different batches of URLs (e.g., `batch1_links.txt`, `batch2_links.txt`).
-   **Training Data:** The SSL training script (`train_ssl.py`) will use whatever data is present in the final processed dataset directory (`data/processed/ssl_dataset/`) that you point it to. If this dataset was generated from a manifest including old and new links, all that data will be used for training.

**Future Consideration:** For more sophisticated management, a future enhancement could involve implementing a database or status file to track processed URLs and their inclusion in specific datasets or training runs. For now, manual management of the URL list is recommended.

---

Further sections on SSL training, ASR fine-tuning, etc., will be added as the project progresses.
