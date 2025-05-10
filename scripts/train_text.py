import os
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(data_dir, max_length=512):
    """
    Load and prepare the dataset for training.
    Assumes the data is in a single text file named 'train.txt' in the data directory.
    """
    train_file = os.path.join(data_dir, "train.txt")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found at {train_file}")
    
    # Load the dataset
    logger.info(f"Loading dataset from {train_file}")
    dataset = load_dataset("text", data_files={"train": train_file})
    
    # Split into train and validation
    dataset = dataset["train"].train_test_split(test_size=0.1)
    
    # Tokenize the dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Tokenizing dataset with max_length={max_length}")
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
        # Set the labels same as inputs for language modeling
        outputs["labels"] = outputs["input_ids"].clone()
        return outputs
    
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_datasets, tokenizer

def train_model(tokenized_datasets, tokenizer, output_dir):
    """
    Train the GPT-2 model on the tokenized dataset.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    
    # Define basic training arguments that should work with most versions
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=3,
    )
    
    logger.info("Training arguments initialized successfully!")
    
    # Initialize the Trainer with only the essential parameters
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, trainer
if __name__ == "__main__":
    # Paths
    DATA_DIR = "data/processed"
    OUTPUT_DIR = "models/text"
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    tokenized_datasets, tokenizer = load_and_prepare_data(DATA_DIR)
    
    # Train the model
    logger.info("Training the model...")
    model, trainer = train_model(tokenized_datasets, tokenizer, OUTPUT_DIR)
    
    logger.info(f"Model training complete. Model saved to {OUTPUT_DIR}")