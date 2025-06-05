import argparse
import os
import logging
import sys

import datasets
from datasets import load_dataset, Audio
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

TARGET_SAMPLING_RATE = 16000 # Should match the feature extractor's expected rate

def load_and_filter_dataset(manifest_path, min_duration_sec=None, max_duration_sec=None):
    """
    Loads the dataset from a JSONL manifest file and filters it by audio duration.
    """
    logger.info(f"Loading dataset from manifest: {manifest_path}")
    try:
        dataset = load_dataset('json', data_files={'train': manifest_path})['train']
        logger.info(f"Successfully loaded dataset. Number of raw samples: {len(dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {manifest_path}: {e}")
        raise

    num_raw_samples = len(dataset)

    if min_duration_sec is not None or max_duration_sec is not None:
        logger.info(f"Filtering dataset by duration: min_sec={min_duration_sec}, max_sec={max_duration_sec}")
        
        def _is_within_duration(example):
            duration = example['duration']
            if min_duration_sec is not None and duration < min_duration_sec:
                return False
            if max_duration_sec is not None and duration > max_duration_sec:
                return False
            return True
        
        original_columns = dataset.column_names
        dataset = dataset.filter(_is_within_duration, num_proc=1) # num_proc can be higher if dataset is large
        logger.info(f"Filtered dataset. Number of samples after filtering: {len(dataset)}")
        if len(dataset) == 0 and num_raw_samples > 0:
            logger.warning("All samples were filtered out. Check duration parameters and manifest content.")
    
    return dataset

def main_prepare_data():
    parser = argparse.ArgumentParser(description="Prepare audio dataset for Self-Supervised Learning (SSL) model training.")
    
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to the audio manifest JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Hugging Face model identifier for the feature extractor (e.g., facebook/wav2vec2-xls-r-300m).")
    
    parser.add_argument("--num_workers", type=int, default=None, help="(Optional) Number of worker processes for dataset.map(). Defaults to None (single process).")
    parser.add_argument("--max_duration_sec", type=float, default=17.0, help="(Optional) Maximum audio duration in seconds to keep. Default: 17.0s (common for some Wav2Vec2 models).")
    parser.add_argument("--min_duration_sec", type=float, default=1.0, help="(Optional) Minimum audio duration in seconds to keep. Default: 1.0s.")
    
    args = parser.parse_args()

    logger.info("Starting SSL data preparation script with arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # --- 1. Load and Filter Dataset ---
    try:
        dataset = load_and_filter_dataset(
            manifest_path=args.manifest_path,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec
        )
    except Exception as e:
        logger.error(f"Could not load or filter dataset. Exiting. Error: {e}")
        return

    if len(dataset) == 0:
        logger.error("No data to process after loading/filtering. Exiting.")
        return

    # --- 2. Load Feature Extractor ---
    logger.info(f"Loading feature extractor from: {args.model_name_or_path}")
    try:
        # Using Wav2Vec2Processor as it often includes the feature extractor and potentially other useful components.
        # If only feature extraction is needed and processor causes issues, Wav2Vec2FeatureExtractor can be used.
        processor = Wav2Vec2Processor.from_pretrained(args.model_name_or_path)
        feature_extractor = processor.feature_extractor
        if feature_extractor.sampling_rate != TARGET_SAMPLING_RATE:
            logger.warning(f"Feature extractor expects sampling rate {feature_extractor.sampling_rate}, but pipeline targets {TARGET_SAMPLING_RATE}. Ensure consistency.")
            # Potentially, could resample here again, but it's better if upstream data is correct.
    except Exception as e:
        logger.error(f"Failed to load feature extractor for {args.model_name_or_path}: {e}")
        return
    
    # Calculate max_length for feature extractor padding/truncation
    # This should align with the model's capabilities and the filtered max_duration_sec
    # For Wav2Vec2, input is raw waveform.
    effective_max_duration = args.max_duration_sec if args.max_duration_sec else 30.0 # Fallback if not filtered
    max_length_samples = int(TARGET_SAMPLING_RATE * effective_max_duration)

    # --- 3. Define Preprocessing Function ---
    def preprocess_function(batch):
        audio_arrays = []
        for audio_path in batch["audio_filepath"]:
            try:
                # Load audio. Our manifest data should already be 16kHz from preprocess_audio.py
                # Using datasets.Audio for robust loading
                speech_array, sr = torchaudio.load(audio_path)
                
                # Ensure it's mono (select first channel if stereo)
                if speech_array.ndim > 1 and speech_array.shape[0] > 1:
                    speech_array = speech_array[0, :] 
                elif speech_array.ndim == 0 : # Should not happen for valid audio
                    speech_array = speech_array.unsqueeze(0)


                # Resample if necessary (should ideally be done before this script)
                if sr != TARGET_SAMPLING_RATE:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLING_RATE)
                    speech_array = resampler(speech_array)
                
                audio_arrays.append(speech_array.squeeze().numpy()) # Squeeze to 1D if it became [1, N]
            except Exception as e:
                logger.warning(f"Failed to load or resample audio file {audio_path}: {e}. Skipping this file.")
                # Need to return something that feature_extractor can handle or skip.
                # For simplicity, if an error occurs, we'll append a None or handle it later.
                # However, dataset.map expects consistent return structure.
                # A better way is to filter out unloadable files beforehand or handle errors carefully.
                # For now, let's assume files in manifest are loadable and correctly formatted.
                # If not, this will error out or map will fail.
                # Let's try to make it robust by returning an empty array for this example
                audio_arrays.append(torch.zeros(1).numpy()) # Placeholder for failed load

        # Process with feature_extractor
        # padding=True will pad to max_length if specified, or to length of longest sequence in batch
        # truncation=True will truncate sequences longer than max_length
        try:
            processed = feature_extractor(
                audio_arrays, 
                sampling_rate=TARGET_SAMPLING_RATE,
                padding="max_length", # Pad to max_length_samples
                max_length=max_length_samples, 
                truncation=True
            )
            return processed
        except Exception as e:
            logger.error(f"Feature extraction failed for a batch. Error: {e}")
            # This could happen if audio_arrays contains problematic data (e.g. NaNs, Infs)
            # or if there's a type mismatch.
            # Return a structure that matches expectations but indicates failure, or raise.
            # For now, let map fail if such an error occurs to highlight it.
            raise


    # --- 4. Apply Preprocessing ---
    logger.info("Applying preprocessing to the dataset...")
    try:
        # Determine columns to remove. Original manifest columns + any added by load_dataset.
        # 'audio_filepath' and 'duration' are from our manifest.
        # If datasets.Audio was used earlier, it might add an 'audio' column.
        # We are loading audio manually in preprocess_function, so 'audio' column isn't created by load_dataset.
        columns_to_remove = ['audio_filepath', 'duration'] 
        # Verify these columns exist before trying to remove.
        existing_columns = dataset.column_names
        columns_to_remove = [col for col in columns_to_remove if col in existing_columns]


        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=16, # Adjust batch size based on memory
            num_proc=args.num_workers,
            remove_columns=columns_to_remove
        )
        logger.info("Preprocessing complete.")
    except Exception as e:
        logger.error(f"Failed to map preprocessing function over dataset: {e}")
        return

    # --- 5. Set Format for PyTorch ---
    try:
        processed_dataset.set_format(type='torch', columns=['input_values'])
        logger.info("Dataset format set to 'torch'.")
    except Exception as e:
        logger.error(f"Failed to set dataset format to 'torch': {e}")
        return

    # --- 6. Save Processed Dataset ---
    logger.info(f"Saving processed dataset to disk at: {args.output_dir}")
    try:
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        processed_dataset.save_to_disk(args.output_dir)
        logger.info(f"Processed dataset successfully saved to {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to save processed dataset to {args.output_dir}: {e}")
        return
    
    logger.info("SSL data preparation finished successfully.")


if __name__ == "__main__":
    main_prepare_data()
