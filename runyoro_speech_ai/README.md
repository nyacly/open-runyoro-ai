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
            -   `data/processed/ssl_dataset/`: (Legacy Transformers) Stores the Hugging Face `Dataset` specifically prepared for Self-Supervised Learning (SSL) model training with the old `train_ssl.py`.
            -   `data/processed/asr_dataset/`: (Future use) Will store datasets prepared for Automatic Speech Recognition (ASR) fine-tuning.
        -   `data/manifest/`: Holds manifest files (typically JSONL format) that list audio file paths and their metadata (like duration).
            -   `data/manifest/audio_manifest.jsonl`: (Default) Manifest generated from all segmented audio by `main_ingest.py`.
            -   `data/manifest/sb_ssl_manifest_for_kmeans.json`: (Example) SpeechBrain-formatted JSON manifest for K-means target generation.
            -   `data/manifest/sb_ssl_manifest.json`: (Example) SpeechBrain-formatted JSON manifest used for actual training (might be same as above).
    -   **`data_ingestion/`**: Contains Python scripts responsible for data acquisition and initial preprocessing.
        -   `main_ingest.py`: The main orchestrator script for the data ingestion pipeline.
        -   `download_youtube.py`, `process_local_files.py`, `preprocess_audio.py`: Core component scripts for ingestion.
    -   **`ssl_training/`**: (Legacy Transformers) Scripts and documentation related to the older Hugging Face Transformers SSL model pre-training.
        -   `prepare_ssl_data.py`: (Legacy) Prepares data for the Transformers `train_ssl.py`.
        -   `train_ssl.py`: (Legacy) Script for running SSL training with Transformers.
        -   `README.md`: (Legacy) Documentation specific to Transformers SSL training.
    -   **`speechbrain_ssl_training/`**: Scripts and documentation for SSL model pre-training using SpeechBrain.
        -   `prepare_sb_ssl_data.py`: Converts general manifest to SpeechBrain JSON format.
        -   `generate_kmeans_targets.py`: Generates K-means targets for HuBERT-style SSL.
        -   `train_sb_ssl.py`: Script for running SSL training with SpeechBrain.
        -   `hparams_ssl.yaml`: Hyperparameter file for SpeechBrain SSL training.
        -   `README.md`: Documentation specific to SpeechBrain SSL training.
    -   **`asr_finetune/`**: Scripts for fine-tuning ASR models (manifest creation and CTC training).
    -   **`inference/`**: (Future use) Will house scripts for using trained models to perform transcription.
    -   **`models/`**: Default directory for storing trained model checkpoints and related artifacts.
        -   `models/ssl/`: (Legacy Transformers) For SSL pre-trained models from `ssl_training/`.
        -   `models/speechbrain_ssl/`: For SSL pre-trained models using SpeechBrain.
        *   `models/asr/`: (Future use) For ASR fine-tuned models.
    -   **`scripts/`**: Utility and helper scripts.
    -   **`docs/`**: Detailed documentation.
    -   **`tests/`**: Unit and integration tests.
    -   **`.github/`**: (If using GitHub) Workflows for GitHub Actions.
    -   **`runyoro_env/` or `.venv/`**: Python virtual environment directory.
    -   **`.gitignore`**: Specifies intentionally untracked files.
    -   **`README.md`**: This file.
    -   **`requirements.txt`**: (Goal) Lists Python package dependencies (creation pending resolution of environment limitations).

## Quick Start: Training an SSL Model (HuBERT-style with SpeechBrain) from YouTube Links

This guide provides a step-by-step walkthrough to download audio from YouTube, process it, and start Self-Supervised Learning (SSL) pre-training using SpeechBrain on your local machine (e.g., a MacBook Pro M4).

*(For more in-depth explanations of each stage, please refer to the documentation in `docs/` and the `speechbrain_ssl_training/README.md`.)*

### Step 1: Environment Setup

Before you begin, ensure your environment is correctly set up:

1.  **Clone the Repository:** If you haven't already, clone this project to your local machine.
2.  **Python Virtual Environment:**
    *   Navigate to the project root (`runyoro_speech_ai/`) in your terminal.
    *   Create and activate a Python virtual environment:
        ```bash
        python3 -m venv runyoro_env
        source runyoro_env/bin/activate
        ```
3.  **Install Dependencies:**
    *   **`ffmpeg`**: Install `ffmpeg` (if not already present). On macOS with Homebrew: `brew install ffmpeg`
    *   **Python Packages**: Install required Python libraries. Key packages include:
        *   `speechbrain`
        *   `torch` (ensure MPS support for Apple Silicon, see [PyTorch Get Started](https://pytorch.org/get-started/locally/))
        *   `torchaudio`
        *   `numpy`
        *   `scikit-learn` (for K-means)
        *   `joblib` (for K-means model saving/loading)
        *   `pyyaml` (for hparams)
        *   `yt-dlp`, `pydub` (for data ingestion)
        *   `transformers`, `datasets`, `accelerate` (if also using the legacy Hugging Face SSL pipeline)
        *(Refer to `speechbrain_ssl_training/README.md` and `docs/data_ingestion.md` for more context on dependencies. A consolidated `requirements.txt` is a future goal.)*
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

### Step 3: Run Data Ingestion and Initial Preprocessing

This step uses the `main_ingest.py` script to download audio from your list of YouTube links, convert it to the required audio format (16kHz mono WAV), segment it into smaller clips, and generate an initial manifest file (`audio_manifest.jsonl`).

Execute the following command from the project root directory (`runyoro_speech_ai/`):

```bash
python ./data_ingestion/main_ingest.py \
    --yt-url-file ./data_ingestion/youtube_links.txt \
    --skip-local-ingest  # Use this flag if you are only processing YouTube links
```

-   **Output:**
    -   Downloaded audio: `data/raw/youtube_downloads/`.
    -   Converted 16kHz mono WAVs: `data/processed/converted_audio_16khz_mono/`.
    -   Segmented audio clips: `data/processed/segmented_audio/`.
    -   Initial JSONL manifest: `data/manifest/audio_manifest.jsonl`.

### Step 4: Prepare Data for SpeechBrain SSL Training

This involves two sub-steps:

#### Step 4a: Convert Manifest for SpeechBrain

Convert the `audio_manifest.jsonl` (from Step 3) into a SpeechBrain-compatible JSON format.

Execute the `prepare_sb_ssl_data.py` script (from project root):
```bash
python ./speechbrain_ssl_training/prepare_sb_ssl_data.py \
    --input_manifest_jsonl ./data/manifest/audio_manifest.jsonl \
    --output_sb_json ./data/manifest/sb_ssl_manifest_for_kmeans.json
```
-   **Output:** Creates `sb_ssl_manifest_for_kmeans.json` in `data/manifest/`.

#### Step 4b: Generate K-means Targets for HuBERT-style SSL

Extract features from the audio (listed in `sb_ssl_manifest_for_kmeans.json`), train a K-means model, and save frame-level cluster labels. These labels are the targets for the HuBERT-style SSL model.

Execute the `generate_kmeans_targets.py` script (from project root):
```bash
# Define where K-means artifacts will be saved; this path will be referenced in hparams.yaml
KMEANS_EXPERIMENT_FOLDER="./models/speechbrain_ssl/kmeans_generation_run1/"
mkdir -p "$KMEANS_EXPERIMENT_FOLDER/kmeans_frame_labels/" # Ensure label directory exists

python ./speechbrain_ssl_training/generate_kmeans_targets.py \
    --hparams_file ./speechbrain_ssl_training/hparams_ssl.yaml \
    --input_sb_manifest_json ./data/manifest/sb_ssl_manifest_for_kmeans.json \
    --output_kmeans_model_path "${KMEANS_EXPERIMENT_FOLDER}/kmeans_model.joblib" \
    --output_target_label_dir "${KMEANS_EXPERIMENT_FOLDER}/kmeans_frame_labels/" \
    --n_clusters 100 \
    # --device "mps" # Or "cuda", "cpu"
```
-   **Output:**
    -   K-means model: `${KMEANS_EXPERIMENT_FOLDER}/kmeans_model.joblib`.
    -   Frame-level labels: `.npy` files in `${KMEANS_EXPERIMENT_FOLDER}/kmeans_frame_labels/`.
-   **Important:** Update your `speechbrain_ssl_training/hparams_ssl.yaml` file:
    -   Set `kmeans_model_path: !ref <output_folder>/kmeans_model.joblib`
    -   Set `target_label_dir: !ref <output_folder>/kmeans_frame_labels/`
    -   Ensure `num_ssl_clusters` matches `--n_clusters` used above.
    -   The `output_folder` in `hparams_ssl.yaml` (or overridden via CLI for `train_sb_ssl.py`) should point to this `$KMEANS_EXPERIMENT_FOLDER` for the K-means artifacts to be found if using `!ref <output_folder>`. Alternatively, use absolute paths or paths relative to `data_folder`. For simplicity, the `test_run_sb_ssl.sh` script modifies a temporary hparams file to point to the correct dummy locations. For a real run, ensure these paths are correctly set in the hparams file used by `train_sb_ssl.py`.

### Step 5: Start SpeechBrain SSL Model Training (HuBERT-style)

This step uses the `train_sb_ssl.py` script to perform HuBERT-style SSL pre-training. The model learns to predict the K-means cluster IDs for masked portions of the audio features.

Execute the `train_sb_ssl.py` script (from project root):
```bash
# Ensure hparams_ssl.yaml is configured with correct paths to K-means model and target labels,
# and other HuBERT-specific parameters.
TRAIN_OUTPUT_FOLDER="./models/speechbrain_ssl/my_runyoro_hubert_ssl_model_run1/"

python ./speechbrain_ssl_training/train_sb_ssl.py \
    --hparams_file ./speechbrain_ssl_training/hparams_ssl.yaml \
    --output_folder "$TRAIN_OUTPUT_FOLDER" \
    --data_folder ./data \
    --number_of_epochs 10 \        # Adjust as needed
    --batch_size 2 \              # Start small (1 or 2) for MacBook M4, adjust based on memory
    # --device "mps" # Or "cuda", "cpu" 
```
-   **Important for MacBook M4 users:**
    -   Use a small `--batch_size`.
    -   Rely on `grad_accumulation_factor` in `hparams_ssl.yaml` for larger effective batch sizes.
    -   Ensure PyTorch is using MPS.
    -   Monitor system resources.

### Step 6: Monitoring Training & Outputs

-   **Console Logs:** Training progress (loss, learning rate, etc.) will be printed to your terminal by SpeechBrain.
-   **Model Checkpoints:** Checkpoints are saved periodically in subdirectories within your specified `--output_folder` (e.g., `$TRAIN_OUTPUT_FOLDER/CKPT+epoch-X.../`) as configured in `hparams_ssl.yaml`.
-   **Final Model:** The final trained SSL model components will be available in the `output_folder`.

### Managing YouTube Links and Re-processing

(This section remains as it was, as it's still relevant for the initial data gathering.)
When working with lists of YouTube links for data collection, consider the following:
... (rest of the section unchanged) ...

---

Further sections on SSL training, ASR fine-tuning, etc., will be added as the project progresses.
