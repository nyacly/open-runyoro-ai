#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting SpeechBrain HuBERT-style SSL Pre-training Test Run..."

# Ensure script is run from the project root for consistent relative paths
echo "Setting Project Root to /app/runyoro_speech_ai/ for consistency in paths..."
cd /app/runyoro_speech_ai/ || exit 1 # Exit if cd fails

echo "Current Working Directory: $(pwd)"


# --- Define Paths ---
HPARAMS_FILE="./speechbrain_ssl_training/hparams_ssl.yaml"
OUTPUT_DIR="./models/speechbrain_ssl/test_run_sb_hubert_ssl_output/" # Changed output dir slightly for clarity
DATA_FOLDER_CLI_OVERRIDE="./data" # Relative to project root for train_sb_ssl.py --data_folder arg

# Path for the dummy SB manifest (relative to project root for creation, referenced via data_folder in hparams)
DUMMY_SB_MANIFEST_DIR_IN_DATA_FOLDER="manifest" # This is 'manifest' inside the DATA_FOLDER_CLI_OVERRIDE
DUMMY_SB_MANIFEST_FILENAME="sb_ssl_manifest_hubert_test_sample.json"
FULL_DUMMY_SB_MANIFEST_PATH="${DATA_FOLDER_CLI_OVERRIDE}/${DUMMY_SB_MANIFEST_DIR_IN_DATA_FOLDER}/${DUMMY_SB_MANIFEST_FILENAME}"

# Path for dummy K-means targets (relative to project root for creation, referenced via data_folder in hparams)
DUMMY_KMEANS_TARGET_DIR_IN_DATA_FOLDER="kmeans_targets_sample" # This is 'kmeans_targets_sample' inside DATA_FOLDER_CLI_OVERRIDE
FULL_DUMMY_KMEANS_TARGET_DIR="${DATA_FOLDER_CLI_OVERRIDE}/${DUMMY_KMEANS_TARGET_DIR_IN_DATA_FOLDER}"
DUMMY_KMEANS_LABEL_FILENAME="dummy_utt_id_001_kmeans_labels.npy" # Must match utt_id in dummy manifest

# --- Create Dummy Data & Directories ---
mkdir -p "$(dirname "${FULL_DUMMY_SB_MANIFEST_PATH}")"
mkdir -p "$FULL_DUMMY_KMEANS_TARGET_DIR"
mkdir -p "$OUTPUT_DIR"

# Create dummy SpeechBrain manifest JSON
echo '{
    "dummy_utt_id_001": {
        "wav": "placeholder/dummy_audio_for_test_run.wav", 
        "duration": 1.0
    }
}' > "${FULL_DUMMY_SB_MANIFEST_PATH}"
echo "Dummy SpeechBrain manifest created at ${FULL_DUMMY_SB_MANIFEST_PATH}"

# Create dummy K-means target label file using Python one-liner
# These labels (0, 1) mean num_ssl_clusters should be at least 2.
python3 -c "import numpy as np; np.save('${FULL_DUMMY_KMEANS_TARGET_DIR}/${DUMMY_KMEANS_LABEL_FILENAME}', np.array([0,1,0,1,0], dtype=np.int32))"
echo "Dummy K-means target label file created at ${FULL_DUMMY_KMEANS_TARGET_DIR}/${DUMMY_KMEANS_LABEL_FILENAME}"

echo "Note: This test primarily checks script startup, hparams loading, and Brain initialization for HuBERT-style SSL."

# --- Temporarily Modify hparams_ssl.yaml ---
TEMP_HPARAMS_FILE="./speechbrain_ssl_training/temp_hparams_for_hubert_test.yaml"

# Start with a copy of the original hparams file
cp "$HPARAMS_FILE" "$TEMP_HPARAMS_FILE"

# Modify specific lines using sed. This is fragile; a proper YAML tool would be better.
# 1. Update train_sb_manifest_file
sed -i "s|^\( *train_sb_manifest_file: *\).*|\1!ref <data_folder>/${DUMMY_SB_MANIFEST_DIR_IN_DATA_FOLDER}/${DUMMY_SB_MANIFEST_FILENAME}|" "$TEMP_HPARAMS_FILE"
# 2. Update target_label_dir
sed -i "s|^\( *target_label_dir: *\).*|\1!ref <data_folder>/${DUMMY_KMEANS_TARGET_DIR_IN_DATA_FOLDER}/|" "$TEMP_HPARAMS_FILE"
# 3. Update kmeans_model_path to a dummy value (it's not loaded by train_sb_ssl.py)
sed -i "s|^\( *kmeans_model_path: *\).*|\1!ref <output_folder>/dummy_kmeans_model.joblib|" "$TEMP_HPARAMS_FILE"
# 4. Update num_ssl_clusters
sed -i "s|^\( *num_ssl_clusters: *\).*|\12|" "$TEMP_HPARAMS_FILE" # Consistent with dummy labels [0,1,0,1,0]
# 5. Ensure ssl_objective_type is hubert_masked_prediction
sed -i "s|^\( *ssl_objective_type: *\).*|\1\"hubert_masked_prediction\"|" "$TEMP_HPARAMS_FILE"


echo "Temporarily modified hparams file created at $TEMP_HPARAMS_FILE with HuBERT-style settings."

# --- Training Parameters for a Very Short Run ---
NUMBER_OF_EPOCHS_OVERRIDE=1
BATCH_SIZE_OVERRIDE=1
# For HuBERT, we need a few steps to ensure masking and loss computation happens.
# Max steps for trainer can be set via CLI in SpeechBrain recipes, but train_sb_ssl.py doesn't have that arg.
# We rely on small dataset (1 item) and 1 epoch.

PYTHON_SCRIPT_PATH="./speechbrain_ssl_training/train_sb_ssl.py"
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "ERROR: Training script not found at $PYTHON_SCRIPT_PATH"
    rm "$TEMP_HPARAMS_FILE" # Clean up
    exit 1
fi

echo "Executing $PYTHON_SCRIPT_PATH with test parameters using $TEMP_HPARAMS_FILE..."

python3 "$PYTHON_SCRIPT_PATH" \
    --hparams_file "$TEMP_HPARAMS_FILE" \
    --output_folder "$OUTPUT_DIR" \
    --data_folder "$DATA_FOLDER_CLI_OVERRIDE" \
    --number_of_epochs "$NUMBER_OF_EPOCHS_OVERRIDE" \
    --batch_size "$BATCH_SIZE_OVERRIDE" \
    # --device "cpu" # Uncomment if GPU/MPS causes issues in test environment

EXIT_CODE=$?

# --- Restore by removing temporary hparams file ---
rm "$TEMP_HPARAMS_FILE"
echo "Temporary hparams file $TEMP_HPARAMS_FILE removed."

if [ $EXIT_CODE -ne 0 ]; then
    echo "SpeechBrain HuBERT-style SSL Test Run FAILED with exit code $EXIT_CODE."
    exit $EXIT_CODE
else
    echo "SpeechBrain HuBERT-style SSL Test Run script finished basic execution check."
    echo "Check logs in $OUTPUT_DIR. Minimal training steps were run."
fi

echo "Script completed."
