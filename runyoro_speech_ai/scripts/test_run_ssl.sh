#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Navigate to the project root if the script is not run from there
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# PROJECT_ROOT="$(dirname "$SCRIPT_DIR")" # Assumes scripts/ is one level down
# cd "$PROJECT_ROOT"
# Note: Worker environment might have specific CWD, adjust if needed or assume CWD is project root /app/runyoro_speech_ai
# For the sandbox, we assume the CWD is /app/, so paths should be relative to that or absolute.
# The train_ssl.py and other scripts are in /app/runyoro_speech_ai/
# Data paths are also typically relative to /app/runyoro_speech_ai/

echo "Setting Project Root to /app/runyoro_speech_ai/ for consistency in paths..."
cd /app/runyoro_speech_ai/ || exit 1 # Exit if cd fails

echo "Current Working Directory: $(pwd)"
echo "Starting SSL Pre-training Test Run..."

# Define paths (relative to the new CWD: /app/runyoro_speech_ai/)
PROCESSED_DATASET_PATH="./data/processed/ssl_dataset_sample/" # A small sample dataset for testing
BASE_MODEL="facebook/wav2vec2-xls-r-300m" # Using a smaller, faster-loading model might be better if available, e.g. a tiny random one
# However, the task specifies xls-r-300m for train_ssl.py, so we use that.
OUTPUT_DIR="./models/ssl/test_run_ssl_output/"

echo "Attempting to create a placeholder for the processed dataset..."
mkdir -p "$PROCESSED_DATASET_PATH"
# Create a dummy dataset_info.json and an empty .arrow file for a basic dataset structure
# This is a very minimal mock of what a Hugging Face dataset saved by save_to_disk looks like.
echo '{"features": {"input_values": {"dtype": "float32", "_type": "Sequence", "feature": {"dtype": "float32", "_type": "Value"}}}, "num_rows": 0}' > "$PROCESSED_DATASET_PATH/dataset_info.json"
mkdir -p "$PROCESSED_DATASET_PATH/train" # Assuming 'train' split if it's a DatasetDict
echo '{"features": {"input_values": {"dtype": "float32", "_type": "Sequence", "feature": {"dtype": "float32", "_type": "Value"}}}, "num_rows": 0}' > "$PROCESSED_DATASET_PATH/train/dataset_info.json"
touch "$PROCESSED_DATASET_PATH/train/dataset.arrow" # An empty Arrow file
# Also create state.json which is typically present
echo '{"_fingerprint": "dummyfp", "_format_columns": null, "_format_kwargs": {}, "_format_type": null, "_output_all_columns": false, "_split": null}' > "$PROCESSED_DATASET_PATH/train/state.json"


echo "Placeholder for processed dataset created at $PROCESSED_DATASET_PATH"
echo "Note: This test primarily checks script startup, argument parsing, and initial model loading."
echo "Full data loading and training functionality depends on a valid dataset from prepare_ssl_data.py."

# Training parameters for a very short run
NUM_EPOCHS=1
BATCH_SIZE=1 
GRAD_ACCUM_STEPS=1
LEARNING_RATE=1e-6 # Smaller LR for tiny test
LOGGING_STEPS=1
SAVE_STEPS=2 # Save at least once if max_steps allows
MAX_STEPS=3  # Run for a minimal number of steps (e.g., 2-3 to test one optimization step)

# Ensure the Python scripts being called are found relative to the new CWD
# train_ssl.py is in ssl_training/ which is a subdir of the CWD /app/runyoro_speech_ai/
PYTHON_SCRIPT_PATH="./ssl_training/train_ssl.py" 

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "ERROR: Training script not found at $PYTHON_SCRIPT_PATH"
    echo "Please ensure the path is correct relative to $(pwd)"
    exit 1
fi

echo "Executing $PYTHON_SCRIPT_PATH with test parameters..."

python3 "$PYTHON_SCRIPT_PATH" \
    --processed_dataset_path "$PROCESSED_DATASET_PATH" \
    --model_name_or_path "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_steps 1 \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 1 \
    --fp16 \
    --seed 42 \
    --dataloader_num_workers 0 \
    --mask_time_prob 0.05 \
    --mask_time_length 10 \
    --mask_feature_prob 0.0 \
    --mask_feature_length 10 \
    --max_steps "$MAX_STEPS" # Crucial for a short test run

echo "SSL Pre-training Test Run script finished."
echo "Check logs and outputs in $OUTPUT_DIR if training proceeded."
echo "Note: If 'dataset_info.json' and dummy arrow file were not enough, the script might have failed at dataset loading or during training."
echo "A true end-to-end test requires running 'prepare_ssl_data.py' on a small data sample first."
# Add a final success message if all commands completed due to set -e
echo "Script completed successfully."
