import sys
import torch
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataio import read_audio
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2 # To load base model
# We might need a specific SSL loss, e.g., for masked prediction
# from speechbrain.nnet.losses import ??? (e.g. some form of reconstruction or prediction loss)
# Or implement a custom one.

import os
import logging
import argparse
import yaml # For loading hparams

logger = logging.getLogger(__name__)
# Basic logging configuration, will be overridden by SpeechBrain's setup if run via its entry points.
# However, for direct script execution, this is useful.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Define custom Brain class for SSL
class SSLBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig # sig is a common SpeechBrain key for audio signals
        
        # Get features from Wav2Vec2 model
        # The HuggingFaceWav2Vec2 lobe from SpeechBrain can act as an encoder
        features = self.modules.wav2vec2_model(wavs, wav_lens) # (B, T, C)

        # SSL Objective: Masked Feature Prediction (conceptual)
        # This part is highly dependent on the chosen SSL strategy and SpeechBrain's tools.
        # 1. Create masked version of features (or input wavs if preferred for some strategies)
        #    SpeechBrain has utilities for SpecAugment, which can be a form of masking.
        #    Or, implement custom masking.
        #    `self.hparams.masking_prob`, `self.hparams.mask_length` could be in YAML.
        #
        #    Example (very simplified masking):
        #    masked_features, mask_indices = self.mask_features(features, wav_lens)
        #
        # 2. If the model has a prediction head for masked features (like BERT for MLM),
        #    pass masked_features through it.
        #    If not, the 'features' themselves (from unmasked input) might be the target
        #    for a reconstruction loss from the masked input passed through encoder again.
        #
        # For now, let's assume 'features' are what we want to predict/reconstruct or use for contrastive loss.
        # The actual SSL mechanism needs to be detailed here based on chosen strategy.
        # This might involve projecting features, quantizing (for Wav2Vec2 original objective), etc.
        
        # For this skeleton, let's just return the features.
        # The actual SSL output would be predictions for masked parts or contrastive elements.
        return features 

    # def mask_features(self, features, wav_lens):
    #     # Placeholder for actual feature masking logic
    #     # This would use parameters from hparams.yaml
    #     # For example, randomly mask some time steps or feature dimensions
    #     # Returns: masked_features, true_masked_values (or indices)
    #     logger.warning("Feature masking not fully implemented in this skeleton.")
    #     return features, None 

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        # This is where the SSL loss is calculated.
        # E.g., if 'predictions' are the output of the model trying to reconstruct masked features,
        # and 'targets' are the true features of the masked parts.
        #
        # For Wav2Vec2 original SSL:
        #  - Contrastive loss between context representations (from masked regions) and quantized true features.
        #  - Diversity loss for codebook usage.
        #
        # For a HuBERT-like masked prediction SSL:
        #  - Cross-entropy loss between predicted cluster IDs and true cluster IDs for masked frames.
        #
        # This skeleton needs a concrete SSL loss function.
        # For now, a dummy loss:
        # loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # A more plausible placeholder if predictions are features and we want to reconstruct them
        # from a hypothetical masked input (this assumes 'predictions' are from a masked input
        # and 'targets' are the original features).
        # This is a very simplified reconstruction loss idea.
        
        # Actual SSL loss logic is critical and complex.
        # For now, let's assume 'predictions' are the final output we are working with.
        # We need a target. If we had 'masked_indices' and 'original_features' from batch:
        # loss = self.hparams.compute_cost(predictions[masked_indices], original_features[masked_indices])

        # This part is highly dependent on what compute_forward returns and what the SSL strategy is.
        # If compute_forward just returns features, we can't compute a loss without more structure.
        # This highlights that the YAML and Brain class need to be tightly coupled
        # with a chosen SSL strategy (e.g. using a specific SpeechBrain SSL model/template).
        
        # Let's create a placeholder loss that just encourages activations to be small
        # This is NOT a real SSL objective but makes the script runnable.
        loss = torch.mean(predictions.pow(2)) 
        
        if stage != sb.Stage.TRAIN:
            # Log additional metrics if needed for validation stage
            # self.val_metrics.append(loss.item()) # Example
            pass
        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch."""
        if stage != sb.Stage.TRAIN:
            # self.val_metrics = [] # Example: Initialize metrics list for validation
            pass

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            logger.info(f"Epoch {epoch}: Training Loss = {stage_loss:.4f}")
        if stage == sb.Stage.VALID:
            # avg_val_loss = sum(self.val_metrics) / len(self.val_metrics) if self.val_metrics else 0
            # logger.info(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.4f}")
            logger.info(f"Epoch {epoch}: Validation Loss (from stage_loss) = {stage_loss:.4f}") # Using stage_loss directly
        if stage == sb.Stage.TEST:
            logger.info(f"Epoch {epoch}: Test Loss = {stage_loss:.4f}")


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    
    logger.info("Preparing datasets...")

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_path): # wav_path is the value from "wav" key in JSON
        # Load audio
        # SpeechBrain's PaddedBatch handles waveform length differences
        try:
            sig = read_audio(wav_path)
            return sig
        except Exception as e:
            logger.error(f"Error reading audio file {wav_path}: {e}")
            # Return a dummy tensor or raise an error to be caught by dataset loader
            # Returning a dummy tensor might hide issues, better to ensure data is clean.
            # For now, let's assume pre-filtering or good data. If not, this will fail.
            # Consider adding a check in the JSON preparation script that all files are loadable.
            raise # Or return sb.signal.dummy_audio(0.1) or similar if recipe handles it.

    # Define datasets
    datasets = {}
    data_info = {
        "train": hparams["train_sb_manifest_file"],
        # Add validation and test manifest if available from hparams
        # "valid": hparams.get("valid_sb_manifest_file"),
        # "test": hparams.get("test_sb_manifest_file"),
    }

    for dataset_name, manifest_file_key in data_info.items():
        if manifest_file_key: # Check if key exists and has a value
            manifest_file = hparams[manifest_file_key] if isinstance(manifest_file_key, str) and manifest_file_key in hparams else manifest_file_key
            if not manifest_file: # If key was present but value was None/empty
                logger.info(f"Manifest file for '{dataset_name}' not provided or empty in hparams. Skipping.")
                continue

            logger.info(f"Loading '{dataset_name}' dataset from: {manifest_file}")
            # Resolve data_folder correctly
            # Assuming hparams["data_folder"] is relative to the hparams file location if not absolute
            # Or, make it relative to CWD if that's more consistent for your setup.
            # For now, SpeechBrain's default usually handles this if paths in JSON are $data_root/
            # And `replacements` is used.
            
            # Ensure the manifest file path is absolute or correctly relative
            # If hparams_file is in 'speechbrain_ssl_training/', and manifest is '../data/manifest/'
            # And data_folder is '../data/'
            # SpeechBrain's default loader should handle replacements like:
            # "wav": "$data_root/processed/segmented_audio/some_file.wav" -> "../data/processed/segmented_audio/some_file.wav"
            # when data_folder = "../data"

            datasets[dataset_name] = DynamicItemDataset.from_json(
                json_path=manifest_file,
                replacements={"data_root": hparams.get("data_folder")}, # Pass data_folder for $data_root replacement
                dynamic_items=[audio_pipeline],
                output_keys=["id", "sig"], # id is the utterance ID from JSON key
            )
            logger.info(f"'{dataset_name}' dataset loaded. Number of samples: {len(datasets[dataset_name])}")
        else:
            logger.info(f"Manifest file key for '{dataset_name}' not found or None in hparams. Skipping {dataset_name} dataset.")

    
    # Sort by duration if specified in hparams (optional)
    # if hparams.get("sort_by_duration", False) and "train" in datasets:
    #    logger.info("Sorting train dataset by duration (descending).")
    #    datasets["train"] = datasets["train"].filtered_sorted(sort_key="duration", reverse=True)

    if not datasets or "train" not in datasets:
        raise ValueError("Training data could not be loaded. Please check 'train_sb_manifest_file' and 'data_folder' in hparams.")

    return datasets

# Main function for training
if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="SpeechBrain SSL Model Training")
    parser.add_argument(
        "--hparams_file",
        type=str,
        required=True,
        help="Path to the hyperparameter YAML file (e.g., hparams_ssl.yaml).",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder where checkpoints and logs will be stored.",
    )
    parser.add_argument(
        "--data_folder", # This will override data_folder in YAML if provided
        type=str,
        help="Path to the main data directory (e.g., ../data/). Overrides 'data_folder' in hparams if specified.",
    )
    # Add common CLI overrides for hparams
    parser.add_argument("--number_of_epochs", type=int, help="Override number of epochs from hparams.")
    parser.add_argument("--batch_size", type=int, help="Override batch size from hparams.")
    parser.add_argument("--lr_adam", type=float, help="Override Adam learning rate from hparams.")
    parser.add_argument("--lr_wav2vec2", type=float, help="Override Wav2Vec2 learning rate from hparams.")
    parser.add_argument("--device", type=str, help="Specify device (e.g., 'cuda', 'mps', 'cpu'). Overrides auto-detection.")


    args = parser.parse_args()

    # Load hyperparameters file with command-line overrides
    # SpeechBrain's `run_on_main` decorator handles this in typical recipes.
    # For direct script execution, we do it manually.
    hparams_file_path = args.hparams_file
    
    # Convert CLI args to a dictionary for overrides, removing None values
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ["hparams_file"]} # output_folder is handled by create_experiment_directory

    # Prepend `../` to paths in hparams if they are relative to the hparams file itself
    # This makes paths like `../data` work correctly if the script is run from `speechbrain_ssl_training`
    # and the hparams file is also there.
    # However, SpeechBrain's `load_extended_yaml` with `sb.create_experiment_directory` usually handles this.
    # For now, assume paths in YAML are relative to hparams file or absolute.
    # `data_folder` in YAML is critical. If it's `../data`, it should point from hparams location.

    with open(hparams_file_path) as fin:
        hparams = sb.load_extended_yaml(fin, overrides=cli_overrides)

    # Ensure output folder exists (SpeechBrain utility)
    # output_folder is taken from args.output_folder, then from hparams if not in args
    # The create_experiment_directory will use args.output_folder if provided.
    # It will also save the hparams file there.
    sb.create_experiment_directory(
        experiment_directory=args.output_folder, # This is where output_folder from args is used
        hyperparams_to_save=hparams_file_path, # Save the original hparams file
        overrides=cli_overrides, # Log overrides
    )
    
    # The `data_folder` in hparams needs to be correctly set up relative to where the script expects it.
    # If `data_folder` in YAML is `../data`, and `hparams_ssl.yaml` is in `speechbrain_ssl_training`,
    # this means `data_folder` resolves to `runyoro_speech_ai/data`.
    # The paths in the manifest (e.g., `sb_ssl_manifest.json`) should then use `$data_root`
    # which will be replaced by this `data_folder`.
    # Example manifest path in YAML: `!ref <data_folder>/manifest/sb_ssl_manifest.json`

    # Prepare datasets
    datasets = dataio_prepare(hparams)

    # Determine device
    run_opts = {"device": sb.core.auto_device()} # Default auto-detection
    if args.device: # If user specified device via CLI
        run_opts["device"] = args.device
    logger.info(f"Running on device: {run_opts['device']}")


    # Initialize Brain object
    ssl_brain = SSLBrain(
        modules=hparams["modules"],
        opt_class=lambda params: getattr(torch.optim, hparams["optimizer"].capitalize())(params, lr=hparams["lr_adam"]),
        hparams=hparams,
        run_opts=run_opts, 
        checkpointer=hparams["checkpointer"],
    )

    # Start training
    # SpeechBrain's PaddedBatch is used by default for DynamicItemDataset
    # Dataloader options can be passed via train_loader_kwargs in fit()
    train_dataloader_opts = hparams.get("train_dataloader_opts", {})
    if "batch_size" not in train_dataloader_opts: # Ensure batch_size from hparams is used
        train_dataloader_opts["batch_size"] = hparams["batch_size"]
    if "num_workers" not in train_dataloader_opts and "num_workers" in hparams: # For SpeechBrain >=0.5.13
         train_dataloader_opts["num_workers"] = hparams.get("num_workers", 0)


    logger.info(f"Starting training with effective batch size: {hparams['batch_size']} and grad_accum: {hparams['grad_accumulation_factor']}")
    
    # Ensure the 'epoch_counter' is available in hparams or use a default starting epoch.
    # SpeechBrain's `Brain.fit` expects an iterable of epochs or an epoch counter.
    # Here, we use the `EpochCounter` from hparams, typically initialized to 0 or loaded from a checkpoint.
    # If `epoch_counter` is not in hparams, create one.
    if "epoch_counter" not in hparams:
        hparams["epoch_counter"] = sb.utils.epoch_loop.EpochCounter(limit=hparams["number_of_epochs"])


    ssl_brain.fit(
        epoch_counter=hparams["epoch_counter"], 
        train_set=datasets["train"],
        train_loader_kwargs=train_dataloader_opts,
        # valid_loader_kwargs=hparams.get("valid_dataloader_opts",{}), # If validation data is present
        # valid_set=datasets.get("valid") 
    )
    
    # Save final model explicitly (optional, as checkpointer might do this)
    # This method is useful if you want to save with a specific name like "final.pt"
    # ssl_brain.save_checkpoint(name="final_model") # This saves the *entire* training state.
    # To save just model weights, you might need:
    # torch.save(ssl_brain.modules.state_dict(), os.path.join(args.output_folder, "final_model_weights.pt"))

    logger.info(f"SSL training with SpeechBrain finished. Checkpoints and logs in: {args.output_folder}")
