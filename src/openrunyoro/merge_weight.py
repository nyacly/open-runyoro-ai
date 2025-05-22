import argparse
import logging
import json
import random
import os
import sys
from rapidfuzz import fuzz

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# File handler for logging
file_handler = logging.FileHandler(os.path.join(log_dir, "merge_weight.log"), mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler for logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def main():
    """
    Main function to parse arguments, initialize random seed, and prepare for merging and weighting chat files.
    """
    parser = argparse.ArgumentParser(description="Merge and weight two JSONL chat files, then split into train/validation sets.")
    parser.add_argument(
        "--primary",
        required=True,
        help="Path to the primary JSONL chat file (e.g., data/processed/book_chat.jsonl).",
    )
    parser.add_argument(
        "--secondary",
        required=True,
        help="Path to the secondary JSONL chat file (e.g., data/processed/web_chat.jsonl).",
    )
    parser.add_argument(
        "--weight",  # Changed from --weight_primary as per instruction clarification
        type=int,
        default=3,
        help="Integer, how many times to sample the primary file (default 3).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proportion of data to set aside for validation (default 0.1).",
    )
    parser.add_argument(
        "--out_train",
        required=True,
        help="Path for the output training JSONL file (e.g., data/train/train_mix.jsonl).",
    )
    parser.add_argument(
        "--out_val",
        required=True,
        help="Path for the output validation JSONL file (e.g., data/train/val_mix.jsonl).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Integer, for deterministic shuffling (default 42).",
    )

    args = parser.parse_args()

    logger.info(f"Parsed arguments: \n"
                f"  Primary file: '{args.primary}'\n"
                f"  Secondary file: '{args.secondary}'\n"
                f"  Weight for primary: {args.weight}\n"
                f"  Validation split: {args.val_split}\n"
                f"  Output train file: '{args.out_train}'\n"
                f"  Output validation file: '{args.out_val}'\n"
                f"  Seed: {args.seed}")

    # Initialize random number generator
    random.seed(args.seed)
    logger.info(f"Random seed initialized with {args.seed}.")

    primary_lines = []
    secondary_lines = []

    # Read Primary File
    try:
        logger.info(f"Reading primary file: {args.primary}")
        with open(args.primary, "r", encoding="utf-8") as f:
            primary_lines = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(primary_lines)} conversations from primary file '{args.primary}'.")
    except FileNotFoundError:
        logger.error(f"Primary file not found: {args.primary}")
        return
    except Exception as e:
        logger.error(f"Error reading primary file '{args.primary}': {e}")
        return

    # Read Secondary File
    try:
        logger.info(f"Reading secondary file: {args.secondary}")
        with open(args.secondary, "r", encoding="utf-8") as f:
            secondary_lines = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(secondary_lines)} conversations from secondary file '{args.secondary}'.")
    except FileNotFoundError:
        logger.error(f"Secondary file not found: {args.secondary}")
        return
    except Exception as e:
        logger.error(f"Error reading secondary file '{args.secondary}': {e}")
        return

    # Weight Primary Data (by repetition)
    if args.weight < 0:
        logger.warning(f"Weight for primary file ({args.weight}) is negative. Treating as 0.")
        weighted_primary_lines = []
    else:
        weighted_primary_lines = primary_lines * args.weight
    logger.info(f"Primary data weighted by factor {args.weight}, resulting in {len(weighted_primary_lines)} primary conversations for combining.")

    # Combine Data
    combined_data = weighted_primary_lines + secondary_lines # Concatenate lists
    logger.info(f"Combined data: {len(weighted_primary_lines)} (weighted primary) + {len(secondary_lines)} (secondary) = {len(combined_data)} total conversations before shuffling.")

    # Deterministic Shuffle
    logger.info("Shuffling combined data...")
    random.shuffle(combined_data)
    logger.info(f"Total {len(combined_data)} conversations in combined_data after shuffling.")

    # Near-Duplicate Removal
    # Using a threshold of 90 for fuzz.ratio (0-100 scale),
    # which means strings are 90% similar.
    # This corresponds to the idea of Levenshtein distance <= 0.1 (similarity >= 0.9).
    if combined_data: # Only run if there's data
        logger.info(f"Starting near-duplicate removal. Initial count: {len(combined_data)} conversations.")
        # The remove_near_duplicates function expects a list of JSON strings.
        # Our combined_data is already a list of strings (JSON objects per line).
        deduplicated_data = remove_near_duplicates(combined_data, threshold_ratio=90.0)
        logger.info(f"Finished near-duplicate removal. Retained {len(deduplicated_data)} conversations from {len(combined_data)}.")
        combined_data = deduplicated_data # Update combined_data to the deduplicated version
    else:
        logger.info("Skipping near-duplicate removal as combined_data is empty.")


    # Train/Validation Split
    if not combined_data:
        logger.warning("No data available after potential deduplication. Cannot create train/val splits.")
        # Create empty output files as per typical behavior? Or just exit?
        # For now, will create empty files if this path is hit and args.out_train/val are specified.
        open(args.out_train, 'w').close()
        open(args.out_val, 'w').close()
        logger.info(f"Created empty output files: '{args.out_train}' and '{args.out_val}'.")
        sys.stderr.write("Training set size: 0 conversations\n")
        sys.stderr.write("Validation set size: 0 conversations\n")
        return

    num_total = len(combined_data)
    num_val = int(num_total * args.val_split)
    num_train = num_total - num_val

    logger.info(f"Splitting data: {num_train} for training, {num_val} for validation.")

    # Data is already shuffled.
    validation_set = combined_data[:num_val]
    training_set = combined_data[num_val:]

    logger.info(f"Training set size: {len(training_set)} conversations.")
    logger.info(f"Validation set size: {len(validation_set)} conversations.")

    # Write Output Files
    try:
        logger.info(f"Writing training set to: {args.out_train}")
        with open(args.out_train, "w", encoding="utf-8") as f_train:
            for conversation_json_str in training_set:
                f_train.write(conversation_json_str + "\n")
        logger.info("Successfully wrote training set.")

        logger.info(f"Writing validation set to: {args.out_val}")
        with open(args.out_val, "w", encoding="utf-8") as f_val:
            for conversation_json_str in validation_set:
                f_val.write(conversation_json_str + "\n")
        logger.info("Successfully wrote validation set.")
    except IOError as e:
        logger.error(f"Error writing output files: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during output file writing: {e}")
        return

    # Print class counts (sizes) to stderr
    sys.stderr.write(f"Training set size: {len(training_set)} conversations\n")
    sys.stderr.write(f"Validation set size: {len(validation_set)} conversations\n")
    
    logger.info("Processing complete.")


def remove_near_duplicates(conversations_json_list: list[str], threshold_ratio: float = 90.0) -> list[str]:
    """
    Removes near-duplicate conversations based on the similarity of their assistant responses.
    A conversation is a JSON string.
    threshold_ratio is on a 0-100 scale (like rapidfuzz.fuzz.ratio).
    """
    deduplicated_conversations_json = []
    # Store only the assistant text of retained conversations for comparison to avoid repeated json.loads()
    retained_assistant_texts = [] 

    for conv_idx, current_conv_json_str in enumerate(conversations_json_list):
        if (conv_idx + 1) % 5000 == 0: # Log progress every 5000 conversations
            logger.info(f"Deduplication progress: Processed {conv_idx + 1}/{len(conversations_json_list)} conversations. "
                        f"Retained: {len(deduplicated_conversations_json)}")
        
        try:
            current_conv_obj = json.loads(current_conv_json_str)
        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON string during deduplication: {current_conv_json_str[:100]}...")
            continue # Skip this malformed conversation

        current_assistant_text = ""
        for turn in current_conv_obj:
            if isinstance(turn, dict) and turn.get("role") == "assistant" and isinstance(turn.get("content"), str):
                current_assistant_text += turn["content"] + " " # Concatenate with space
        current_assistant_text = current_assistant_text.strip()

        if not current_assistant_text: # No assistant text, or only non-string content
            deduplicated_conversations_json.append(current_conv_json_str)
            # No specific assistant text to store for comparison, or it's empty.
            # We could store an empty string or a special marker if we want to prevent
            # other no-assistant-text convos from accumulating, but for now, just add.
            retained_assistant_texts.append("") 
            continue

        is_near_duplicate = False
        # Compare with assistant text of already retained conversations
        # This is O(N*M) where N is total, M is current number of retained. Can be slow.
        # For large M, process.extractOne might be faster if it can be adapted, 
        # but it finds the *best* match, not just *any* match above threshold.
        # A direct loop is more accurate for "is it a duplicate of *any* existing".
        for existing_assistant_text in retained_assistant_texts:
            if not existing_assistant_text: # Skip comparison if an existing retained one had no assistant text
                continue
            
            # Calculate similarity ratio (0-100)
            similarity = fuzz.ratio(current_assistant_text, existing_assistant_text)
            if similarity > threshold_ratio:
                is_near_duplicate = True
                # logger.debug(f"Near duplicate found! Ratio: {similarity:.2f}%\n  New: {current_assistant_text[:100]}...\n  Old: {existing_assistant_text[:100]}...")
                break
        
        if not is_near_duplicate:
            deduplicated_conversations_json.append(current_conv_json_str)
            retained_assistant_texts.append(current_assistant_text)
            
    return deduplicated_conversations_json

if __name__ == "__main__":
    main()
