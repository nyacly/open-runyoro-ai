import os
import sys
import torch
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataio import read_audio
from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from sklearn.cluster import MiniBatchKMeans # Efficient for large data
import numpy as np
import joblib # For saving sklearn models
import logging
import argparse
import yaml # For loading hparams (to get wav2vec2_hub, data_folder, etc.)
import hyperpyyaml
import json # For reading the SB manifest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def dataio_prepare_for_feature_extraction(hparams, manifest_file):
    """Prepares dataset for feature extraction."""
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = read_audio(wav)
        return sig

    dataset = DynamicItemDataset.from_json(
        json_path=manifest_file,
        replacements={"data_root": hparams["data_folder"]},
        dynamic_items=[audio_pipeline],
        output_keys=["id", "sig"],
    )
    return dataset

def extract_features(dataset, wav2vec2_model, device, hparams):
    """Extracts features from the dataset using the wav2vec2 model."""
    logger.info("Starting feature extraction...")
    all_features_list = [] # List to store features for each utterance
    all_utt_ids = []       # List to store utterance IDs in order of processing
    
    wav2vec2_model.eval() # Ensure model is in eval mode
    
    # Determine which layer to extract features from (e.g., from hparams)
    # Default to last layer if not specified, but intermediate layers are common for HuBERT targets
    # This parameter is conceptual for this script; actual layer extraction logic may need HuggingFaceWav2Vec2 modification or direct HF model usage.
    extraction_layer = hparams.get("kmeans_extraction_layer", None) 
    if extraction_layer is not None:
        logger.warning(f"kmeans_extraction_layer={extraction_layer} is specified, but current script uses default HuggingFaceWav2Vec2 output (likely last hidden state). Advanced layer selection requires model modification or specific SpeechBrain lobe features.")


    with torch.no_grad():
        for i, batch in enumerate(dataset): # Assuming dataset yields individual samples or small batches
            utt_id = batch["id"]
            all_utt_ids.append(utt_id)
            
            # Ensure wavs is 2D [batch_size, num_samples] for wav2vec2_model
            wavs = batch["sig"]
            if wavs.ndim == 1: # If it's a single sample, add batch dimension
                wavs = wavs.unsqueeze(0)
            wavs = wavs.to(device)
            
            # wav_lens should be relative (0.0 to 1.0)
            # For a single utterance, its relative length is 1.0
            wav_lens = torch.tensor([1.0] * wavs.shape[0], device=device) 

            try:
                feats = wav2vec2_model(wavs, wav_lens) # (B, T, C), B=1 for single sample iteration
                
                # Ensure feats is not empty and is 2D (Time, Channels) before appending
                # Squeeze batch dimension if B=1
                if feats.ndim == 3 and feats.shape[0] == 1:
                    current_utt_features = feats.squeeze(0).cpu().numpy()
                elif feats.ndim == 2: # Already (T,C)
                    current_utt_features = feats.cpu().numpy()
                else:
                    logger.warning(f"Unexpected feature shape for {utt_id}: {feats.shape}. Skipping.")
                    all_features_list.append(np.array([]).reshape(0, feats.shape[-1] if feats.ndim > 1 else 1)) # Keep structure for alignment
                    continue

                if current_utt_features.shape[0] == 0: # If utterance results in 0 frames
                     logger.warning(f"Utterance {utt_id} resulted in 0 feature frames. Skipping.")
                     all_features_list.append(np.array([]).reshape(0, current_utt_features.shape[-1] if current_utt_features.ndim > 1 else 1))
                else:
                    all_features_list.append(current_utt_features)

            except Exception as e:
                logger.error(f"Error extracting features for utterance {utt_id}: {e}. Appending empty features.", exc_info=True)
                # Attempt to get feature dimension if possible, otherwise use a placeholder like 1
                # This ensures the per_file_features list has an entry for every ID, even if empty
                # So that predict_and_save_labels doesn't misalign.
                try:
                    # Try to get a dummy feature dim if the model is loaded
                    dummy_dim = wav2vec2_model.model.config.hidden_size # Or similar attribute
                    all_features_list.append(np.array([]).reshape(0, dummy_dim))
                except:
                    all_features_list.append(np.array([]).reshape(0, 1)) # Fallback, might cause issues later

            if (i + 1) % 100 == 0:
                logger.info(f"Extracted features for {i + 1} utterances.")
    
    logger.info(f"Feature extraction complete for {len(all_features_list)} utterances.")
    
    # Filter out any utterances that resulted in empty features before concatenating for K-Means
    # But keep track of which ones were valid for K-Means input
    features_for_kmeans = [f for f in all_features_list if f.ndim == 2 and f.shape[0] > 0]
    
    if not features_for_kmeans:
        logger.error("No valid features extracted to train K-means. All utterances might have been too short or failed processing.")
        return None, all_features_list, all_utt_ids # Still return all_features_list and all_utt_ids for potential partial label saving

    concatenated_features = np.concatenate(features_for_kmeans, axis=0)
    logger.info(f"Shape of concatenated features for K-means: {concatenated_features.shape}")
    return concatenated_features, all_features_list, all_utt_ids

def train_kmeans(features, n_clusters, random_state, batch_size_kmeans):
    """Trains MiniBatchKMeans model."""
    if features is None or features.shape[0] == 0:
        logger.error("Cannot train K-means: No features provided.")
        return None
        
    logger.info(f"Starting K-means training with {n_clusters} clusters on {features.shape[0]} frames...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size_kmeans, 
        verbose=1, # Log progress from MiniBatchKMeans
        n_init='auto' # Let sklearn decide based on batch_size and n_clusters
    )
    kmeans.fit(features)
    logger.info("K-means training complete.")
    return kmeans

def predict_and_save_labels(kmeans_model, per_file_features_list, dataset_ids, output_target_dir):
    """Predicts cluster labels for each frame and saves them."""
    if kmeans_model is None:
        logger.error("K-means model not available. Skipping label prediction.")
        return

    logger.info("Predicting and saving K-means cluster labels...")
    ensure_output_dir(output_target_dir)
    
    if len(dataset_ids) != len(per_file_features_list):
        logger.error(f"Critical Error: Mismatch in number of IDs ({len(dataset_ids)}) and number of feature sets ({len(per_file_features_list)}). Cannot save labels correctly.")
        return

    saved_count = 0
    for i, utt_id in enumerate(dataset_ids):
        features_for_utt = per_file_features_list[i]
        
        if not isinstance(features_for_utt, np.ndarray) or features_for_utt.ndim != 2 or features_for_utt.shape[0] == 0:
            logger.warning(f"Skipping label prediction for {utt_id} due to empty or malformed features (shape: {getattr(features_for_utt, 'shape', 'N/A')}). This utterance might have failed feature extraction.")
            continue

        try:
            labels = kmeans_model.predict(features_for_utt) # (Num_Frames_in_Utterance,)
            label_path = os.path.join(output_target_dir, f"{utt_id}_kmeans_labels.npy")
            np.save(label_path, labels.astype(np.int32)) # Save as numpy array
            saved_count += 1
            if saved_count % 100 == 0: # Log every 100 successfully saved labels
                logger.info(f"Saved labels for {saved_count} utterances (current: {utt_id}).")
        except Exception as e:
            logger.error(f"Error predicting or saving labels for {utt_id}: {e}", exc_info=True)

    logger.info(f"K-means labels prediction and saving finished. Successfully saved labels for {saved_count}/{len(dataset_ids)} utterances.")

def ensure_output_dir(directory_path):
    # Ensure directory path is not empty or just a filename
    if directory_path and directory_path != os.path.basename(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    elif not directory_path:
        logger.warning("Attempted to ensure output directory, but path was empty.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K-means targets for HuBERT-style SSL pre-training.")
    parser.add_argument("--hparams_file", type=str, required=True, help="Path to SpeechBrain hyperparameter YAML file (e.g., hparams_ssl.yaml).")
    parser.add_argument("--input_sb_manifest_json", type=str, required=True, help="Path to the SpeechBrain JSON manifest (output of prepare_sb_ssl_data.py).")
    parser.add_argument("--output_kmeans_model_path", type=str, required=True, help="Path to save the trained K-means model (e.g., kmeans_model.joblib).")
    parser.add_argument("--output_target_label_dir", type=str, required=True, help="Directory to save the predicted K-means frame labels (e.g., data/kmeans_targets/).")
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of K-means clusters.")
    parser.add_argument("--kmeans_random_state", type=int, default=42, help="Random state for K-means.")
    parser.add_argument("--kmeans_batch_size", type=int, default=10000, help="Batch size for MiniBatchKMeans.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu'). Defaults to auto-detection by SpeechBrain.")
    
    args = parser.parse_args()

    kmeans_experiment_folder = os.getenv("KMEANS_EXPERIMENT_FOLDER")
    if not kmeans_experiment_folder:
        # It's good practice to log or raise an error if critical env var is missing
        logger.error("KMEANS_EXPERIMENT_FOLDER environment variable not set. Please set it before running the script.")
        sys.exit(1) # Make sure sys is imported if you use sys.exit(1)
    
    overrides_dict = {"output_folder": kmeans_experiment_folder}
    # Load hparams to get data_folder, wav2vec2_hub etc.
    with open(args.hparams_file) as fin:
        hparams = hyperpyyaml.load_hyperpyyaml(fin, overrides=overrides_dict, overrides_must_match=False)

    # Device handling
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Check for MPS (Apple Silicon)
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # Ensure output directories exist before starting
    ensure_output_dir(os.path.dirname(args.output_kmeans_model_path))
    ensure_output_dir(args.output_target_label_dir)

    # Prepare dataset
    logger.info(f"Loading dataset from: {args.input_sb_manifest_json}")
    dataset = dataio_prepare_for_feature_extraction(hparams, args.input_sb_manifest_json)
    
    # Initialize Wav2Vec2 model from HuggingFace via SpeechBrain
    model_cache_dir = os.path.join(hparams.get("output_folder", "model_cache"), "wav2vec2_hf_cache") # output_folder from hparams for main training
    ensure_output_dir(model_cache_dir) # Ensure cache dir exists
    logger.info(f"Loading Wav2Vec2 model: {hparams['wav2vec2_hub']}. Cache path: {model_cache_dir}")
    
    wav2vec2_model = Wav2Vec2(
        source=hparams["wav2vec2_hub"],
        save_path=model_cache_dir, 
        output_norm=hparams.get("wav2vec2_output_norm", True),
        freeze_feature_extractor=hparams.get("wav2vec2_freeze_feature_extractor", True), # Usually True for feature extraction
        # output_all_hiddens=True # Set this if you want to select intermediate layers from hparams
    ).to(device)
    
    # Extract features
    concatenated_features, per_file_features_list, dataset_ids = extract_features(dataset, wav2vec2_model, device, hparams)
    
    if concatenated_features is None or concatenated_features.shape[0] == 0:
        logger.error("No features were extracted. Cannot proceed with K-means training. Check audio files, paths, and feature extraction logs.")
        sys.exit(1)

    # Train K-means
    kmeans_model = train_kmeans(concatenated_features, args.n_clusters, args.kmeans_random_state, args.kmeans_batch_size)
    
    if kmeans_model is None:
        logger.error("K-means model training failed. Cannot proceed to save model or predict labels.")
        sys.exit(1)

    # Save K-means model
    joblib.dump(kmeans_model, args.output_kmeans_model_path)
    logger.info(f"K-means model saved to {args.output_kmeans_model_path}")

    # Predict and save labels
    predict_and_save_labels(kmeans_model, per_file_features_list, dataset_ids, args.output_target_label_dir)

    logger.info("K-means target generation finished.")
