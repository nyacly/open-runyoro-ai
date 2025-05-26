# Self-Supervised Learning (SSL) Model Training with SpeechBrain

This section outlines the pipeline for SSL pre-training using the SpeechBrain toolkit, focusing on a **HuBERT-style masked prediction objective**. Our goal is to adapt a multilingual pre-trained speech model (e.g., `facebook/wav2vec2-xls-r-300m`) to Runyoro audio by training it to predict discrete acoustic units (derived from K-means clustering) for masked portions of the audio.

## Rationale for SpeechBrain

Due to persistent and unusual difficulties in setting up a stable Hugging Face Transformers environment on the target Mac M4 system, SpeechBrain is being adopted as a robust alternative for speech processing tasks.

## Chosen Approach for SSL Model and Base Checkpoint

-   **Base Model Target:** `facebook/wav2vec2-xls-r-300m`. The encoder part of this model will be used as the feature extractor.
-   **Loading Mechanism:** SpeechBrain's `speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2` lobe is used to load the pre-trained weights from the Hugging Face model.
-   **Training Goal:** Continued pre-training using a HuBERT-style masked prediction objective. The model learns to predict K-means derived discrete acoustic units for masked segments of the audio features.

## Key Dependencies

-   `speechbrain`
-   `torch` (with MPS support for Apple Silicon)
-   `torchaudio`
-   `numpy`
-   `scikit-learn` (for `MiniBatchKMeans`)
-   `joblib` (for saving/loading the K-means model)
-   `soundfile`, `pydub` (for audio data handling in earlier stages)
-   `pyyaml` (for hparams)
-   Other dependencies will be managed via `pip`.

## Data Preparation for HuBERT-style SSL Training

The data preparation now involves three main steps to get from raw audio to data suitable for HuBERT-style training:

1.  **Initial Audio Processing (External to this directory):**
    *   Use the main data ingestion pipeline (`runyoro_speech_ai/data_ingestion/main_ingest.py`) to download/copy raw audio, convert it to 16kHz mono WAV, segment it, and generate an initial manifest file (e.g., `../data/manifest/audio_manifest.jsonl`).
    *   Refer to `docs/data_ingestion.md` for details on this stage.

2.  **Convert Manifest to SpeechBrain JSON Format (`prepare_sb_ssl_data.py`):**
    *   This script converts the `.jsonl` manifest (from step 1) into a single JSON file that SpeechBrain recipes typically use.
    *   **Command (from project root `runyoro_speech_ai/`):**
        ```bash
        python ./speechbrain_ssl_training/prepare_sb_ssl_data.py \
            --input_manifest_jsonl ./data/manifest/audio_manifest.jsonl \
            --output_sb_json ./data/manifest/sb_ssl_manifest_for_kmeans.json 
            # Consider naming it specifically for this stage if you keep multiple manifests
        ```
    *   **Output:** Creates `sb_ssl_manifest_for_kmeans.json` (or your chosen name) in `data/manifest/`.

3.  **Generate K-means Targets (`generate_kmeans_targets.py`):**
    *   This script extracts features from the audio files listed in the SpeechBrain JSON manifest (from step 2), trains a K-means model on these features, and then predicts and saves frame-level cluster assignments (pseudo-labels) for each audio file.
    *   **Inputs:**
        *   The SpeechBrain JSON manifest (e.g., `sb_ssl_manifest_for_kmeans.json`).
        *   An `hparams_ssl.yaml` file (to get `wav2vec2_hub` for feature extraction and `data_folder`).
    *   **Outputs:**
        *   A trained K-means model file (e.g., `kmeans_model.joblib`).
        *   A directory containing `.npy` files, where each file stores the frame-level K-means labels for one utterance.
    *   **Command (from project root `runyoro_speech_ai/`):**
        ```bash
        # Ensure hparams_ssl.yaml exists and data_folder within it is correctly set (e.g. to "./data")
        # Also ensure output_folder is specified for the experiment, as K-means model and targets are saved there.
        # Example: output_folder for this step could be ./models/speechbrain_ssl/kmeans_generation/
        EXPERIMENT_OUTPUT_FOLDER="./models/speechbrain_ssl/kmeans_generation/" 
        mkdir -p "$EXPERIMENT_OUTPUT_FOLDER" # Create if it doesn't exist

        python ./speechbrain_ssl_training/generate_kmeans_targets.py \
            --hparams_file ./speechbrain_ssl_training/hparams_ssl.yaml \
            --input_sb_manifest_json ./data/manifest/sb_ssl_manifest_for_kmeans.json \
            --output_kmeans_model_path "${EXPERIMENT_OUTPUT_FOLDER}/kmeans_model.joblib" \
            --output_target_label_dir "${EXPERIMENT_OUTPUT_FOLDER}/kmeans_frame_labels/" \
            --n_clusters 100 # Or your desired number of clusters
            # --device "mps" # Or "cuda", "cpu"
        ```
    *   **Important:** The paths `output_kmeans_model_path` and `output_target_label_dir` (or their equivalents) will then need to be correctly referenced in your `hparams_ssl.yaml` for the actual SSL training stage. The `hparams_ssl.yaml` uses `!ref <output_folder>/...` for these, so ensure the `output_folder` for the training run points to where these K-means artifacts are saved or copy them appropriately.

## SSL Model Training with SpeechBrain (HuBERT-style)

The SSL training is orchestrated by the `train_sb_ssl.py` script, using hyperparameters defined in `hparams_ssl.yaml`.

1.  **Review Hyperparameters (`hparams_ssl.yaml`):**
    Ensure `hparams_ssl.yaml` is configured for HuBERT-style training:
    *   `train_sb_manifest_file`: Points to the SpeechBrain JSON manifest (e.g., `sb_ssl_manifest_for_kmeans.json`).
    *   `wav2vec2_hub`: Base HF model (e.g., `facebook/wav2vec2-xls-r-300m`).
    *   `kmeans_model_path`: Path to the K-means model file (output of `generate_kmeans_targets.py`).
    *   `target_label_dir`: Path to the directory containing K-means frame label `.npy` files.
    *   `num_ssl_clusters`: Must match the number of clusters used for K-means.
    *   Masking parameters: `mask_prob`, `mask_length`, `mask_min_spans`.
    *   `ssl_prediction_head`: Definition of the linear head to predict K-means cluster IDs.
    *   `ssl_objective_type`: Should be set to `"hubert_masked_prediction"`.
    *   Standard training parameters: `number_of_epochs`, `batch_size`, learning rates, etc.

2.  **Running `train_sb_ssl.py`:**
    The training script now loads audio and its corresponding K-means frame labels. It applies masking to features from the Wav2Vec2 encoder, uses a prediction head to predict cluster IDs for these masked frames, and computes a Cross-Entropy loss.
    From the project root directory (`runyoro_speech_ai/`), execute:
    ```bash
    # Ensure your hparams_ssl.yaml correctly points to the K-means model and target labels directory.
    # The output_folder for this training run will be distinct from the K-means generation output_folder.
    TRAIN_OUTPUT_FOLDER="./models/speechbrain_ssl/my_runyoro_hubert_ssl_model/"

    python ./speechbrain_ssl_training/train_sb_ssl.py \
        --hparams_file ./speechbrain_ssl_training/hparams_ssl.yaml \
        --output_folder "$TRAIN_OUTPUT_FOLDER" \
        --data_folder ./data \ # Ensure this aligns with how paths are structured in your manifest and hparams
        # Optional overrides for hparams:
        # --number_of_epochs 25 
        # --batch_size 2 
    ```
    -   `output_folder`: Specifies where checkpoints, logs, and other training artifacts for *this training run* will be saved.
    -   `data_folder`: Helps resolve paths like `$data_root` in the manifest if used.

3.  **Monitoring Training:**
    -   Logs will be printed to the console.
    -   Checkpoints are saved in the `output_folder` as configured in `hparams_ssl.yaml`.

4.  **Test Run Script:**
    The `test_run_sb_ssl.sh` script has been updated to support a quick test of the HuBERT-style training pipeline (using dummy K-means targets and a modified temporary hparams file).
    From the project root:
    ```bash
    bash ./scripts/test_run_sb_ssl.sh
    ```

**Important Note on SSL Objective & K-means Targets:**
The SSL training now implements a **HuBERT-style masked prediction objective**. The model is trained to predict K-means derived acoustic unit "pseudo-labels" for masked portions of the audio features.

-   **Quality of K-means Targets:** The effectiveness of this SSL pre-training is significantly influenced by the quality of the K-means targets. These targets are generated by clustering features from the pre-trained Wav2Vec2 model. The choice of layer for feature extraction (though `generate_kmeans_targets.py` currently uses the default output layer of the SpeechBrain `HuggingFaceWav2Vec2` lobe) and the number of clusters (`num_ssl_clusters`) are important hyperparameters that can affect target quality.
-   **Iterative Refinement (Advanced):** In the original HuBERT paper, the K-means clustering and subsequent SSL training can be an iterative process. One could train a HuBERT model, then use its (now better) features to re-cluster and generate improved targets for further training. This iterative refinement is not yet implemented in the current script suite but is a potential area for future enhancement.
