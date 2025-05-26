#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting SpeechBrain SSL Pre-training Test Run..."

# Ensure script is run from the project root for consistent relative paths
# The sandbox CWD is /app/, and project root is /app/runyoro_speech_ai/
echo "Setting Project Root to /app/runyoro_speech_ai/ for consistency in paths..."
cd /app/runyoro_speech_ai/ || exit 1 # Exit if cd fails

echo "Current Working Directory: $(pwd)"


# Define paths (these should align with your project structure and hparams_ssl.yaml)
HPARAMS_FILE="./speechbrain_ssl_training/hparams_ssl.yaml"
OUTPUT_DIR="./models/speechbrain_ssl/test_run_sb_ssl_output/"
# data_folder for CLI override in train_sb_ssl.py, relative to project root
DATA_FOLDER_CLI_OVERRIDE="./data" 

# Dummy manifest file for SpeechBrain. This path is relative to the project root.
# The train_sb_ssl.py script will use this via data_folder CLI override + path in hparams.
# We will also update hparams_ssl.yaml temporarily for this test run if needed,
# or rely on the data_folder override to correctly resolve the path specified in hparams.
# The hparams_ssl.yaml has: train_sb_manifest_file: !ref <data_folder>/manifest/sb_ssl_manifest.json
# So, if data_folder (CLI override) is ./data, it will look for ./data/manifest/sb_ssl_manifest.json
# Let's create a sample manifest at that default location for the test.
DUMMY_SB_MANIFEST_PATH_IN_HPARAMS_DEFAULT_LOCATION="./data/manifest/sb_ssl_manifest_test_sample.json"

# Create directories if they don't exist
mkdir -p "$(dirname "${DUMMY_SB_MANIFEST_PATH_IN_HPARAMS_DEFAULT_LOCATION}")" # Ensure data/manifest exists
mkdir -p "$OUTPUT_DIR"

# Create a minimal, valid JSON for SpeechBrain (one dummy entry)
# The "wav" path should be something that won't cause an immediate error if SpeechBrain
# tries to stat it, even if it doesn't exist. A relative path within the expected
# structure is fine for a dummy entry.
# The `prepare_sb_ssl_data.py` script would generate paths based on actual files.
echo '{
    "dummy_utt_id_001": {
        "wav": "placeholder/dummy_audio_for_test_run.wav", 
        "duration": 1.0
    }
}' > "${DUMMY_SB_MANIFEST_PATH_IN_HPARAMS_DEFAULT_LOCATION}"

echo "Dummy SpeechBrain manifest created at ${DUMMY_SB_MANIFEST_PATH_IN_HPARAMS_DEFAULT_LOCATION}"
echo "Note: This test primarily checks script startup, hparams loading, and Brain initialization."
echo "Full data loading functionality depends on running prepare_sb_ssl_data.py first on actual data."
echo "This script will temporarily modify ${HPARAMS_FILE} to use the dummy manifest, then restore it."

# --- Temporarily Modify hparams_ssl.yaml to use the dummy manifest ---
# This is safer than assuming complex path resolution for a simple test script.
ORIGINAL_HPARAMS_CONTENT=$(cat "$HPARAMS_FILE")
# Use basename for the manifest file in YAML, as data_folder will provide the directory context
DUMMY_MANIFEST_BASENAME=$(basename "$DUMMY_SB_MANIFEST_PATH_IN_HPARAMS_DEFAULT_LOCATION")

# Create a temporary hparams file for the test run
TEMP_HPARAMS_FILE="./speechbrain_ssl_training/temp_hparams_for_test.yaml"

# Modify the train_sb_manifest_file line
# This sed command looks for 'train_sb_manifest_file:' and replaces the line.
# It assumes a simple structure for that line.
# Using a Python script for robust YAML editing would be better for complex cases.
sed "s|^\( *train_sb_manifest_file: *\).*|\1!ref <data_folder>/manifest/${DUMMY_MANIFEST_BASENAME}|" "$HPARAMS_FILE" > "$TEMP_HPARAMS_FILE"

echo "Temporarily modified hparams file created at $TEMP_HPARAMS_FILE to use the dummy manifest."


# Training parameters for a very short run (override some from YAML if needed)
# These are passed as CLI args to train_sb_ssl.py which overrides values in the YAML.
NUMBER_OF_EPOCHS_OVERRIDE=1
BATCH_SIZE_OVERRIDE=1
# max_batches or similar for quick stop (SpeechBrain's Brain.fit has `max_epochs` and `max_steps`)
# We can control this via number_of_epochs and very small dataset.


# Check if the python script exists
PYTHON_SCRIPT_PATH="./speechbrain_ssl_training/train_sb_ssl.py"
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "ERROR: Training script not found at $PYTHON_SCRIPT_PATH"
    echo "Please ensure the path is correct relative to $(pwd)"
    # Restore hparams if modified and exit
    rm "$TEMP_HPARAMS_FILE"
    echo "Restored original hparams file (if modified)."
    exit 1
fi


echo "Executing $PYTHON_SCRIPT_PATH with test parameters using $TEMP_HPARAMS_FILE..."

# The train_sb_ssl.py script already has CLI args for number_of_epochs and batch_size.
# It also has a --data_folder override.
python3 "$PYTHON_SCRIPT_PATH" \
    --hparams_file "$TEMP_HPARAMS_FILE" \
    --output_folder "$OUTPUT_DIR" \
    --data_folder "$DATA_FOLDER_CLI_OVERRIDE" \
    --number_of_epochs "$NUMBER_OF_EPOCHS_OVERRIDE" \
    --batch_size "$BATCH_SIZE_OVERRIDE" \
    # Add --device cpu if MPS/CUDA is problematic in test env for quick check
    # --device "cpu" 

EXIT_CODE=$?

# --- Restore original hparams_ssl.yaml ---
rm "$TEMP_HPARAMS_FILE"
echo "Temporary hparams file $TEMP_HPARAMS_FILE removed."


if [ $EXIT_CODE -ne 0 ]; then
    echo "SpeechBrain SSL Pre-training Test Run FAILED with exit code $EXIT_CODE."
    exit $EXIT_CODE
else
    echo "SpeechBrain SSL Pre-training Test Run script finished (basic execution check)."
    echo "Check logs in $OUTPUT_DIR if training proceeded (it would be minimal with dummy data/loss)."
    echo "This test does not validate learning, only that the script runs without critical errors for a few steps."
fi

# Add a final success message if all commands completed due to set -e
echo "Script completed."
