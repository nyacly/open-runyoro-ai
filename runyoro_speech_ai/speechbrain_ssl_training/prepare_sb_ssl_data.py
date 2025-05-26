import argparse
import json
import os
import logging
import sys

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def ensure_output_dir_for_file(filepath):
    """Creates the directory for the given filepath if it doesn't exist."""
    dir_path = os.path.dirname(filepath)
    if dir_path: # Ensure dir_path is not empty (e.g. if output is in current dir)
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Ensured output directory exists for file: {filepath} (Directory: {dir_path})")
        except OSError as e:
            logger.error(f"Could not create output directory {dir_path} for file {filepath}. {e}")
            raise # Re-raise to stop execution if output dir can't be made

def convert_manifest_to_speechbrain_format(input_manifest_jsonl, output_sb_json, data_root=None):
    """
    Converts an audio manifest from JSONL format (one JSON object per line)
    to SpeechBrain's JSON format (a single JSON object with utterance IDs as keys).

    Args:
        input_manifest_jsonl (str): Path to the input JSONL manifest file.
        output_sb_json (str): Path to save the output SpeechBrain JSON manifest.
        data_root (str, optional): A root path to prepend to audio filepaths if they are relative.
                                   If None, paths are made absolute based on their current form.
    """
    logger.info(f"Starting manifest conversion...")
    logger.info(f"Input JSONL manifest: {input_manifest_jsonl}")
    logger.info(f"Output SpeechBrain JSON: {output_sb_json}")
    if data_root:
        logger.info(f"Using data_root: {data_root}")

    speechbrain_data = {}
    entries_processed = 0
    entries_skipped_no_path = 0
    
    try:
        ensure_output_dir_for_file(output_sb_json)
    except Exception:
        logger.error(f"Exiting due to output directory creation failure for {output_sb_json}.")
        return False

    try:
        with open(input_manifest_jsonl, 'r', encoding='utf-8') as f_in:
            for line_number, line in enumerate(f_in, 1):
                try:
                    line = line.strip()
                    if not line:
                        logger.debug(f"Skipping empty line at line {line_number}.")
                        continue
                    
                    entry = json.loads(line)
                    
                    audio_filepath_original = entry.get('audio_filepath')
                    duration = entry.get('duration')

                    if not audio_filepath_original or duration is None:
                        logger.warning(f"Skipping entry at line {line_number} due to missing 'audio_filepath' or 'duration': {line}")
                        entries_skipped_no_path += 1
                        continue

                    # Generate utterance ID from the filename without extension
                    utt_id = os.path.splitext(os.path.basename(audio_filepath_original))[0]
                    
                    # Handle potential duplicate utterance IDs - append a counter
                    # This is a simple way; more robust might involve checking existing keys
                    original_utt_id = utt_id
                    counter = 1
                    while utt_id in speechbrain_data:
                        utt_id = f"{original_utt_id}_{counter}"
                        counter += 1
                    if utt_id != original_utt_id:
                         logger.debug(f"Duplicate utt_id '{original_utt_id}' found. Using '{utt_id}' instead.")


                    # Process filepath
                    if data_root:
                        if os.path.isabs(audio_filepath_original):
                            # If path is already absolute, data_root might be a sanity check or ignored
                            # For now, we assume if path is absolute, it's correct.
                            final_audio_path = audio_filepath_original
                        else:
                            final_audio_path = os.path.join(data_root, audio_filepath_original)
                    else:
                        # Default to making the path absolute if no data_root is provided
                        final_audio_path = os.path.abspath(audio_filepath_original)
                    
                    # Ensure the path exists, or SpeechBrain will fail later
                    if not os.path.exists(final_audio_path):
                        logger.warning(f"Audio file for utt_id '{utt_id}' not found at resolved path: {final_audio_path}. Original: {audio_filepath_original}. Skipping.")
                        entries_skipped_no_path += 1
                        continue

                    speechbrain_data[utt_id] = {
                        "wav": final_audio_path, # Store the processed path
                        "duration": float(duration)
                    }
                    entries_processed += 1
                    if entries_processed % 1000 == 0:
                        logger.info(f"Processed {entries_processed} entries...")

                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line {line_number}: {line.strip()}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_number} ('{line.strip()}'): {e}")
        
        logger.info(f"Finished reading input manifest. Total entries processed: {entries_processed}")
        logger.info(f"Total entries skipped (missing path/duration or file not found): {entries_skipped_no_path}")

    except FileNotFoundError:
        logger.error(f"Input manifest file not found: {input_manifest_jsonl}")
        return False
    except Exception as e:
        logger.error(f"An error occurred while reading {input_manifest_jsonl}: {e}")
        return False

    if not speechbrain_data:
        logger.warning("No data was successfully processed. Output JSON will be empty or not written.")
        # Decide if an empty JSON should be written or not. Let's write it if file could be opened.
        # return False # Or True if an empty JSON is acceptable. For now, let it proceed to write.

    try:
        with open(output_sb_json, 'w', encoding='utf-8') as f_out:
            json.dump(speechbrain_data, f_out, indent=4, ensure_ascii=False)
        logger.info(f"Successfully wrote SpeechBrain JSON manifest to: {output_sb_json}")
    except IOError as e:
        logger.error(f"Could not write to output file {output_sb_json}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while writing {output_sb_json}: {e}")
        return False
        
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL audio manifest to SpeechBrain JSON format.")
    
    parser.add_argument(
        "--input_manifest_jsonl",
        type=str,
        required=True,
        help="Path to the input manifest file (JSONL format, e.g., ../data/manifest/audio_manifest.jsonl)."
    )
    parser.add_argument(
        "--output_sb_json",
        type=str,
        required=True,
        help="Path for the output SpeechBrain JSON manifest (e.g., ../data/manifest/sb_ssl_manifest.json)."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None, # Default to None, meaning paths are made absolute or used as-is if already absolute.
        help="(Optional) A root path to prepend to relative audio filepaths. If not provided, paths from input manifest are made absolute."
    )

    args = parser.parse_args()

    logger.info("Starting SpeechBrain manifest preparation script with arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    if convert_manifest_to_speechbrain_format(
        args.input_manifest_jsonl, 
        args.output_sb_json, 
        args.data_root
    ):
        logger.info("Manifest conversion completed successfully.")
    else:
        logger.error("Manifest conversion failed. Check logs for details.")

if __name__ == "__main__":
    main()
