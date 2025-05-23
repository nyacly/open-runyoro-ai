import argparse
import logging
import os
import sys
import torch

from datasets import load_from_disk
from transformers import (
    Wav2Vec2FeatureExtractor, # Using FeatureExtractor directly as Processor might not be needed for pretraining data collation
    Wav2Vec2ForPreTraining,
    TrainingArguments,
    Trainer,
    DataCollatorForWav2Vec2Pretraining,
    set_seed
)
# Wav2Vec2Processor can be used if tokenizer/text components are also needed,
# but for pre-training on audio features alone, FeatureExtractor is often sufficient.

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def check_mps_or_cuda_availability():
    """Checks for MPS (Apple Silicon GPU) or CUDA availability."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS backend is available on this device.")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("CUDA is available on this device.")
        return "cuda"
    else:
        logger.info("Neither MPS nor CUDA is available. Using CPU.")
        return "cpu"

def main_train_ssl():
    parser = argparse.ArgumentParser(description="Train a Self-Supervised Learning (SSL) model (Wav2Vec2) for continued pre-training.")

    # --- Path Arguments ---
    parser.add_argument("--processed_dataset_path", type=str, required=True, help="Path to the directory containing the processed dataset (output of prepare_ssl_data.py).")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/wav2vec2-xls-r-300m", help="Base model identifier (e.g., facebook/wav2vec2-xls-r-300m).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model, checkpoints, and logs.")

    # --- Training Hyperparameters ---
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    train_group.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training.")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.")
    train_group.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    train_group.add_argument("--warmup_steps", type=int, default=500, help="Number of steps for the warmup phase.")
    train_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to apply.")
    train_group.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    train_group.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    train_group.add_argument("--save_total_limit", type=int, default=2, help="Limit the total number of checkpoints. Deletes the oldest checkpoints.")
    train_group.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision training instead of 32-bit. Requires NVIDIA Apex or native PyTorch AMP.")
    train_group.add_argument("--seed", type=int, default=42, help="Random seed for initialization for reproducibility.")
    train_group.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")


    # --- Data Collator Masking Parameters ---
    mask_group = parser.add_argument_group("Wav2Vec2 Masking Parameters")
    mask_group.add_argument("--mask_time_prob", type=float, default=0.65, help="Probability of masking time steps.") # Default from Wav2Vec2 paper for pretraining
    mask_group.add_argument("--mask_time_length", type=int, default=10, help="Length of masked time spans.")
    mask_group.add_argument("--mask_feature_prob", type=float, default=0.0, help="Probability of masking features (usually 0 for Wav2Vec2 pre-training as it masks in time).")
    mask_group.add_argument("--mask_feature_length", type=int, default=10, help="Length of masked feature spans (if mask_feature_prob > 0).")
    
    # --- Dataloader Parameters ---
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of workers for the dataloader. Default 0 (main process).")


    args = parser.parse_args()

    logger.info("Starting SSL Model Training script with arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    
    # --- Set Seed for Reproducibility ---
    set_seed(args.seed)

    # --- Determine Device ---
    # Note: Trainer handles device placement based on TrainingArguments (e.g. use_mps_device, or CUDA visibility)
    # This check is more for informational purposes or if manual device placement was needed.
    device_type = check_mps_or_cuda_availability()
    logger.info(f"Primary compute device determined: {device_type}")


    # --- 1. Load Processed Dataset ---
    logger.info(f"Loading processed dataset from: {args.processed_dataset_path}")
    try:
        # Assuming the dataset saved by prepare_ssl_data.py is a single Dataset object
        # or a DatasetDict where we are interested in the 'train' split.
        # If it was saved as a single Dataset, load_from_disk will return that.
        # If it was a DatasetDict (e.g. dataset_dict.save_to_disk()), it also works.
        processed_dataset = load_from_disk(args.processed_dataset_path)
        
        # If it's a DatasetDict, and we only have one split (e.g. 'train'), we might use it directly.
        # Or, if multiple splits exist from some prior step, select the one for training.
        # For this script, we assume the processed_dataset_path contains the data intended for training.
        # If it's a DatasetDict, let's assume 'train' is the key or it's the only key.
        if isinstance(processed_dataset, datasets.DatasetDict):
            if 'train' in processed_dataset:
                train_dataset = processed_dataset['train']
            elif len(processed_dataset.keys()) == 1: # If only one split, use it
                train_dataset = processed_dataset[list(processed_dataset.keys())[0]]
            else:
                raise ValueError(f"Processed dataset at {args.processed_dataset_path} is a DatasetDict with multiple splits. Please specify which split to use or ensure only one 'train' split exists.")
        elif isinstance(processed_dataset, datasets.Dataset):
            train_dataset = processed_dataset
        else:
            raise TypeError(f"Loaded dataset is of unexpected type: {type(processed_dataset)}")

        logger.info(f"Successfully loaded dataset. Number of training samples: {len(train_dataset)}")
        logger.info(f"Dataset features: {train_dataset.features}")
    except Exception as e:
        logger.error(f"Failed to load processed dataset from {args.processed_dataset_path}: {e}")
        return

    # --- 2. Load Model and Feature Extractor ---
    logger.info(f"Loading model and feature extractor from: {args.model_name_or_path}")
    try:
        # For pre-training, we typically only need the feature extractor.
        # If the base model comes with a processor that includes a tokenizer, that's fine,
        # but the tokenizer part won't be used by DataCollatorForWav2Vec2Pretraining.
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
        
        model = Wav2Vec2ForPreTraining.from_pretrained(args.model_name_or_path)

        if args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for the model.")
            model.gradient_checkpointing_enable()

    except Exception as e:
        logger.error(f"Failed to load model or feature extractor from {args.model_name_or_path}: {e}")
        return

    # --- 3. Initialize Data Collator ---
    logger.info("Initializing Data Collator for Wav2Vec2 Pretraining.")
    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model, # The model is needed by the collator to know about certain properties like attention mask format
        feature_extractor=feature_extractor,
        padding="longest", # Pad to the longest sequence in the batch. Max length padding done in prepare_ssl_data.py
        mask_time_prob=args.mask_time_prob,
        mask_time_length=args.mask_time_length,
        mask_feature_prob=args.mask_feature_prob,
        mask_feature_length=args.mask_feature_length,
    )
    
    # --- 4. Training Arguments ---
    logger.info("Defining Training Arguments.")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16, # PyTorch AMP will be used if available on CUDA/MPS
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="tensorboard", # Default, can be changed to "wandb" or "none"
        # For pre-training, it's crucial that unused columns are NOT removed by default if the collator needs them.
        # However, our dataset from prepare_ssl_data.py should only have 'input_values' and possibly attention_mask.
        # DataCollatorForWav2Vec2Pretraining will handle the creation of attention_mask from input_values if not present.
        remove_unused_columns=False, 
        # use_mps_device will be automatically set if 'mps' is available and no_cuda is False.
        # Explicitly setting it can be done if needed:
        # use_mps_device = (device_type == "mps") 
    )
    
    # If MPS is available and fp16 is true, PyTorch >=1.13 should handle it.
    # For older versions, fp16 on MPS might be problematic or unsupported.
    if device_type == "mps" and args.fp16:
        logger.info("fp16 requested on MPS. Ensure PyTorch version supports this well (>=1.13+).")
        # Note: `torchrun` or `accelerate launch` might be better for MPS + distributed setups.
        # Trainer handles single device MPS if `PYTORCH_ENABLE_MPS_FALLBACK=1` is not set causing issues.


    # --- 5. Initialize Trainer ---
    logger.info("Initializing Trainer.")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        # eval_dataset can be added if there's a validation set for pre-training (e.g. to monitor perplexity)
    )

    # --- 6. Start Training ---
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training finished successfully.")
    except Exception as e:
        logger.error(f"Training failed with an error: {e}", exc_info=True) # Log traceback
        return

    # --- 7. Save Final Model & Feature Extractor ---
    logger.info(f"Saving final model and feature extractor to {args.output_dir}")
    try:
        trainer.save_model(args.output_dir) # Saves the model state_dict and config
        # Save the feature extractor config alongside the model for easy reloading
        feature_extractor.save_pretrained(args.output_dir)
        logger.info("Model and feature extractor saved.")
    except Exception as e:
        logger.error(f"Failed to save final model or feature extractor: {e}")

    logger.info("SSL Model Training script completed.")

if __name__ == "__main__":
    main_train_ssl()
