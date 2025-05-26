import sys
import torch
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataio import read_audio
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
# import speechbrain.nnet.losses as sb_losses # Not used for now, F.cross_entropy is direct

import os
import logging
import argparse
import yaml # For loading hparams
import numpy as np # For loading .npy K-means targets

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(module)s.%(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SSLBrain(sb.Brain):
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)
        
        # Dynamically set input size for ssl_head if not already set
        if hasattr(self.modules, 'wav2vec2_model') and hasattr(self.modules, 'ssl_head'):
            try:
                # This assumes wav2vec2_model is the SpeechBrain HuggingFaceWav2Vec2 lobe
                # and its underlying model is a standard HuggingFace PreTrainedModel with a config.
                if hasattr(self.modules.wav2vec2_model, 'model') and \
                   hasattr(self.modules.wav2vec2_model.model, 'config') and \
                   hasattr(self.modules.wav2vec2_model.model.config, 'hidden_size'):
                    
                    encoder_out_dim = self.modules.wav2vec2_model.model.config.hidden_size
                    
                    # Check if ssl_head is a SpeechBrain NNET Linear or similar and if input_size needs setting
                    current_ssl_head = self.modules.ssl_head
                    if isinstance(current_ssl_head, sb.nnet.linear.Linear):
                        # SpeechBrain's Linear module stores layers in a Sequential; first layer is the actual nn.Linear
                        if hasattr(current_ssl_head, 'layers') and len(current_ssl_head.layers) > 0 and \
                           isinstance(current_ssl_head.layers[0], torch.nn.Linear):
                            actual_linear_layer = current_ssl_head.layers[0]
                            
                            if actual_linear_layer.in_features != encoder_out_dim :
                                logger.info(f"Dynamically re-initializing ssl_head. Original in_features: {actual_linear_layer.in_features}, New (encoder_out_dim): {encoder_out_dim}")
                                self.modules.ssl_head = sb.nnet.linear.Linear(
                                    input_size=encoder_out_dim, 
                                    n_neurons=self.hparams.num_ssl_clusters, # From hparams
                                    bias=True # Match YAML
                                ).to(self.device)
                            else:
                                logger.info(f"SSL_Head input size ({actual_linear_layer.in_features}) already matches Wav2Vec2 output size ({encoder_out_dim}). No re-initialization needed.")
                        else: # Fallback if structure is different, or if input_size was meant to be set by SpeechBrain from YAML
                             logger.warning("Could not verify/set ssl_head input size dynamically. Assuming it's correctly set if non-zero.")

                else:
                    logger.warning("Could not dynamically determine wav2vec2_model output size for ssl_head. Ensure input_size is correctly set in YAML or manually.")
            except Exception as e:
                logger.warning(f"Error dynamically setting/checking ssl_head input size: {e}. Ensure it's correctly defined in hparams.")
        
        # Initialize learnable mask embedding for HuBERT-style masking
        # This parameter will be part of the model and moved to device by Brain
        if hasattr(self.modules, 'wav2vec2_model') and hasattr(self.modules.wav2vec2_model, 'model'):
             feat_dim = self.modules.wav2vec2_model.model.config.hidden_size
             self.mask_embedding = torch.nn.Parameter(torch.FloatTensor(feat_dim).uniform_())
             logger.info(f"Initialized learnable mask embedding of shape: {self.mask_embedding.shape}")
        else:
            # This case should ideally not happen if modules are correctly defined in YAML and loaded.
            logger.error("wav2vec2_model not available for determining feature dimension for mask_embedding. Masking will fail.")
            # Set a placeholder to prevent AttributeError, but it won't be correct.
            self.mask_embedding = torch.nn.Parameter(torch.FloatTensor(1).uniform_()) # Placeholder

    def mask_acoustic_features(self, features, wav_lens):
        """
        Applies HuBERT-style masking to acoustic features (output of Wav2Vec2 encoder).
        Features: (B, T, C). wav_lens: (B,) relative lengths (0-1).
        Returns: masked_features (B, T, C) and boolean_mask (B, T).
        """
        batch_size, seq_len, _ = features.shape
        
        boolean_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=features.device)
        masked_features = features.clone()

        for i in range(batch_size):
            # Actual number of frames for this item based on relative wav_lens
            # This assumes feature sequence length is proportional to audio length,
            # which is true for Wav2Vec2 like models.
            current_num_frames = int(torch.round(wav_lens[i] * seq_len))
            if current_num_frames == 0:
                continue

            # Number of frames to mask based on mask_prob
            num_frames_to_mask = int(current_num_frames * self.hparams.mask_prob)
            
            # Adjust if less than mask_min_spans * mask_length
            min_frames_for_min_spans = self.hparams.mask_min_spans * self.hparams.mask_length
            if num_frames_to_mask < min_frames_for_min_spans and current_num_frames >= min_frames_for_min_spans :
                num_frames_to_mask = min_frames_for_min_spans
            
            num_frames_to_mask = min(num_frames_to_mask, current_num_frames) # Cap at available frames

            masked_indices_count = 0
            # Iterate to select mask spans. This is a simplified approach.
            # fairseq's HuBERT has a more complex geometric distribution based span selection.
            attempts = 0 # To prevent infinite loops if masking is difficult
            while masked_indices_count < num_frames_to_mask and attempts < current_num_frames * 5:
                span_start = torch.randint(0, current_num_frames - self.hparams.mask_length + 1, (1,)).item()
                
                # Check if this span would make us exceed num_frames_to_mask too much
                # This is a simple way to try and get close to the target number of masked frames
                if masked_indices_count + self.hparams.mask_length > num_frames_to_mask * 1.5 and masked_indices_count > 0: # Allow some overshoot
                    attempts += 1
                    continue 
                
                for j in range(self.hparams.mask_length):
                    idx_to_mask = span_start + j
                    if idx_to_mask < current_num_frames: # Ensure we are within actual frames
                        if not boolean_mask[i, idx_to_mask]: # If not already masked
                            boolean_mask[i, idx_to_mask] = True
                            masked_features[i, idx_to_mask, :] = self.mask_embedding
                            masked_indices_count += 1
                    if masked_indices_count >= num_frames_to_mask: break
                if masked_indices_count >= num_frames_to_mask: break
                attempts += 1
            
            if num_frames_to_mask > 0 and masked_indices_count == 0:
                logger.debug(f"Item {i}: Could not mask any frames. Requested: {num_frames_to_mask}, Actual frames: {current_num_frames}")
            elif masked_indices_count < num_frames_to_mask:
                logger.debug(f"Item {i}: Masked {masked_indices_count}/{num_frames_to_mask} frames. Actual frames: {current_num_frames}")

        return masked_features, boolean_mask


    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # K-means targets are padded by PaddedBatch if lengths vary
        kmeans_targets_padded, kmeans_target_lens_abs = batch.kmeans_targets 
        
        # Get features from Wav2Vec2 model (Encoder output)
        encoder_output_features = self.modules.wav2vec2_model(wavs, wav_lens) # (B, T, D_encoder)

        # Apply HuBERT-style masking to these encoder output features
        # The mask_acoustic_features function uses wav_lens (relative audio lengths)
        # to determine the number of feature frames for each item in the batch.
        masked_input_for_head, time_mask_indices = self.mask_acoustic_features(encoder_output_features, wav_lens)
        
        # Pass the masked encoder output through the SSL prediction head
        predictions_logits = self.modules.ssl_head(masked_input_for_head) # (B, T, num_clusters)
        
        return predictions_logits, time_mask_indices, kmeans_targets_padded, kmeans_target_lens_abs

    def compute_objectives(self, forward_outputs, batch, stage):
        predictions_logits, time_mask_indices, kmeans_targets_padded, kmeans_target_lens_abs = forward_outputs
        
        kmeans_targets_padded = kmeans_targets_padded.to(predictions_logits.device)

        # Align target sequence length with prediction sequence length if necessary
        pred_seq_len = predictions_logits.size(1)
        target_seq_len = kmeans_targets_padded.size(1)

        if pred_seq_len != target_seq_len:
            logger.debug(
                f"Aligning sequence lengths: Pred_len={pred_seq_len}, Target_len={target_seq_len}. Truncating to min length."
            )
            min_len = min(pred_seq_len, target_seq_len)
            predictions_logits = predictions_logits[:, :min_len, :]
            kmeans_targets_padded = kmeans_targets_padded[:, :min_len]
            time_mask_indices = time_mask_indices[:, :min_len]
            # We also need to adjust kmeans_target_lens_abs if we truncate targets
            # This is important if targets had padding that's now removed by truncation.
            # For simplicity, assume that if lengths differ, it's minor and related to conv layers,
            # and that kmeans_target_lens_abs still broadly applies to the (now potentially truncated) targets.
            # A more robust solution would re-calculate target_lens if truncation happens.
            # For now, we assume the mask (time_mask_indices) will correctly select valid regions.

        # Select only the logits and targets at masked positions
        masked_logits = predictions_logits[time_mask_indices] 
        masked_targets = kmeans_targets_padded[time_mask_indices]

        if masked_logits.nelement() == 0 or masked_targets.nelement() == 0:
            logger.warning("No masked frames for loss computation in this batch. Returning zero loss.")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Compute Cross-Entropy loss
        # masked_logits shape: (TotalMaskedFrames, NumClusters)
        # masked_targets shape: (TotalMaskedFrames)
        loss = F.cross_entropy(
            masked_logits.reshape(-1, self.hparams.num_ssl_clusters), 
            masked_targets.reshape(-1),
            # ignore_index: K-means IDs should all be valid (0 to N-1).
            # If K-means target files could have padding values (e.g., -100), specify ignore_index.
            # Assuming for now that loaded K-means targets are only valid IDs for masked positions.
        )
        
        if stage != sb.Stage.TRAIN:
            # Optional: calculate accuracy for monitoring
            with torch.no_grad():
                predicted_ids = torch.argmax(masked_logits, dim=-1)
                correct_predictions = (predicted_ids == masked_targets).sum().item()
                total_masked = masked_targets.numel()
                if total_masked > 0 :
                    self.last_batch_accuracy = correct_predictions / total_masked
                else:
                    self.last_batch_accuracy = 0.0 # Or some other indicator
        return loss

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.accuracies_this_epoch = [] # Store accuracies for averaging

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_name = stage.name.capitalize()
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss # sb.Brain tracks this automatically
            logger.info(f"Epoch {epoch}: Training Loss = {stage_loss:.4f}")
        else: # Validation or Test
            logger.info(f"Epoch {epoch}: {stage_name} Loss = {stage_loss:.4f}")
            if hasattr(self, 'accuracies_this_epoch') and self.accuracies_this_epoch:
                avg_accuracy = sum(self.accuracies_this_epoch) / len(self.accuracies_this_epoch)
                logger.info(f"Epoch {epoch}: {stage_name} Average Accuracy = {avg_accuracy:.3f}")
            elif hasattr(self, 'last_batch_accuracy'): # Log last batch accuracy if list is empty (e.g. only one batch for val)
                 logger.info(f"Epoch {epoch}: {stage_name} Last Batch Accuracy = {self.last_batch_accuracy:.3f}")


def dataio_prepare(hparams):
    logger.info("Preparing datasets for HuBERT-style SSL training...")
    
    # target_label_dir is resolved from hparams.
    # In YAML: target_label_dir: !ref <output_folder>/kmeans_frame_labels/
    # 'output_folder' is set in the main script part from CLI args before hparams are fully parsed for dataio.
    # So, hparams["target_label_dir"] should be the correct, absolute path.
    target_label_dir = hparams["target_label_dir"]
    logger.info(f"K-means target labels will be loaded from: {target_label_dir}")
    if not os.path.isdir(target_label_dir):
        # This check might be too early if output_folder is created by SB's main loop later.
        # However, for data loading, it should ideally exist.
        logger.warning(f"K-means target label directory does not exist: {target_label_dir}. This will likely cause errors during data loading.")

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_path):
        try:
            sig = read_audio(wav_path)
            return sig
        except Exception as e:
            logger.error(f"Error reading audio file {wav_path}: {e}", exc_info=True)
            raise

    @sb.utils.data_pipeline.takes("id") # Takes utterance ID from the JSON manifest key
    @sb.utils.data_pipeline.provides("kmeans_targets")
    def label_pipeline(utt_id):
        label_filename = f"{utt_id}_kmeans_labels.npy"
        label_path = os.path.join(target_label_dir, label_filename)
        try:
            targets = np.load(label_path)
            return torch.from_numpy(targets).long()
        except FileNotFoundError:
            logger.error(f"K-means target file not found: {label_path}. This utterance will fail.")
            raise FileNotFoundError(f"K-means target file not found: {label_path}")
        except Exception as e:
            logger.error(f"Error loading K-means target file {label_path}: {e}", exc_info=True)
            raise

    datasets_map = {}
    data_info = {
        "train": hparams["train_sb_manifest_file"],
        # "valid": hparams.get("valid_sb_manifest_file"), # Add if you have validation
    }

    for dataset_name, manifest_file_ref in data_info.items():
        if manifest_file_ref: # YAML ref like !ref <data_folder>/...
            actual_manifest_file = hparams[manifest_file_ref] if isinstance(manifest_file_ref, str) and manifest_file_ref in hparams else manifest_file_ref
            if not actual_manifest_file: # If the reference resulted in None or empty
                logger.info(f"Manifest file for '{dataset_name}' (from key '{manifest_file_ref}') is not defined or empty. Skipping.")
                continue
            
            logger.info(f"Loading '{dataset_name}' dataset from: {actual_manifest_file}")
            datasets_map[dataset_name] = DynamicItemDataset.from_json(
                json_path=actual_manifest_file,
                replacements={"data_root": hparams.get("data_folder")}, # data_folder from YAML
                dynamic_items=[audio_pipeline, label_pipeline],
                output_keys=["id", "sig", "kmeans_targets"], # Ensure "kmeans_targets" is here
            )
            logger.info(f"'{dataset_name}' dataset loaded. Number of samples: {len(datasets_map[dataset_name])}")
        else:
            logger.info(f"Manifest file reference for '{dataset_name}' not found or None. Skipping.")

    if not datasets_map or "train" not in datasets_map:
        raise ValueError("Training data could not be loaded. Check manifest paths and data_folder in hparams.")

    return datasets_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpeechBrain HuBERT-style SSL Model Training")
    parser.add_argument("--hparams_file", type=str, required=True, help="Path to hparams YAML file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder for checkpoints and logs.")
    parser.add_argument("--data_folder", type=str, help="Path to main data directory (overrides hparams).")
    parser.add_argument("--number_of_epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch_size", type=int, help="Override batch size.")
    parser.add_argument("--device", type=str, help="Specify device (e.g., 'cuda', 'mps', 'cpu').")

    args = parser.parse_args()
    hparams_file_path = args.hparams_file
    
    # CLI overrides dictionary
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ["hparams_file"]}
    # Ensure output_folder from CLI is part of overrides if provided, as it's crucial for !ref resolution
    if args.output_folder:
        cli_overrides["output_folder"] = args.output_folder


    with open(hparams_file_path) as fin:
        hparams = sb.load_extended_yaml(fin, overrides=cli_overrides)
    
    # Important: Ensure hparams['output_folder'] is set to the CLI arg *before* dataio_prepare,
    # as target_label_dir might use !ref <output_folder>.
    # sb.create_experiment_directory does this, but if we resolve paths before, ensure it's set.
    # The `load_extended_yaml` with `overrides` should handle this if `output_folder` is a !ref in YAML.
    # If `output_folder` itself is not a !ref but other things are, like `target_label_dir: !ref <output_folder>/...`,
    # then `output_folder` must be correctly defined in `hparams` when `dataio_prepare` is called.
    # The CLI override for output_folder is the primary way.
    
    sb.create_experiment_directory(
        experiment_directory=args.output_folder, # This is the definitive output_folder
        hyperparams_to_save=hparams_file_path,
        overrides=cli_overrides, # This will also log the final overrides applied
    )
    
    # Update hparams with the definitive output_folder for dataio_prepare, if it wasn't already set via overrides.
    # This ensures that if `target_label_dir` relies on `output_folder` via `!ref`, it resolves correctly.
    hparams["output_folder"] = args.output_folder


    datasets = dataio_prepare(hparams)

    run_opts = {"device": sb.core.auto_device()}
    if args.device:
        run_opts["device"] = args.device
    logger.info(f"Running on device: {run_opts['device']}")

    # Initialize Brain object
    # paramwise_optimizers can be used for differential learning rates if needed
    # For example:
    # if "paramwise_optimizers" not in hparams:
    #    hparams["paramwise_optimizers"] = {
    #        "wav2vec2_model": {"optimizer": torch.optim.Adam, "lr": hparams.get("lr_wav2vec2", hparams["lr_adam"])},
    #        "ssl_head": {"optimizer": torch.optim.Adam, "lr": hparams["lr_adam"]}
    #    }
    # And then pass paramwise_optimizers=hparams["paramwise_optimizers"] to Brain
    
    ssl_brain = SSLBrain(
        modules=hparams["modules"],
        opt_class=lambda params: getattr(torch.optim, hparams["optimizer"].capitalize())(
            params, lr=hparams["lr_adam"] 
        ),
        hparams=hparams,
        run_opts=run_opts, 
        checkpointer=hparams["checkpointer"],
    )

    train_dataloader_opts = hparams.get("train_dataloader_opts", {})
    if "batch_size" not in train_dataloader_opts: # Ensure batch_size from hparams (possibly overridden by CLI) is used
        train_dataloader_opts["batch_size"] = hparams["batch_size"]
    
    # num_workers can be specified in YAML under train_dataloader_opts or globally
    if "num_workers" not in train_dataloader_opts and "num_workers" in hparams:
         train_dataloader_opts["num_workers"] = hparams.get("num_workers", 0)
    elif "num_workers" not in train_dataloader_opts and "dataloader_num_workers" in hparams: # backward compat for older hparam name
         train_dataloader_opts["num_workers"] = hparams.get("dataloader_num_workers", 0)


    logger.info(f"Starting training with effective batch size: {hparams['batch_size']} * {hparams['grad_accumulation_factor']}")
    
    if "epoch_counter" not in hparams:
        hparams["epoch_counter"] = sb.utils.epoch_loop.EpochCounter(limit=hparams["number_of_epochs"])

    ssl_brain.fit(
        epoch_counter=hparams["epoch_counter"], 
        train_set=datasets["train"],
        train_loader_kwargs=train_dataloader_opts,
        # valid_set=datasets.get("valid"), # Uncomment if validation is added
        # valid_loader_kwargs=hparams.get("valid_dataloader_opts", {})
    )
    
    logger.info(f"HuBERT-style SSL training finished. Checkpoints and logs in: {args.output_folder}")
