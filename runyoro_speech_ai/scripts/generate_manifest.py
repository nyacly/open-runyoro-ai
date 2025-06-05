import argparse
import os
import json
import logging
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Configure basic logging - This will be configured by the main_ingest.py if imported.
# If run standalone, this basic config will apply.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def ensure_output_dir_for_file(filepath):
    """Creates the directory for the given filepath if it doesn't exist."""
    dir_path = os.path.dirname(filepath)
    if dir_path: 
        try:
            os.makedirs(dir_path, exist_ok=True)
            logging.debug(f"Ensured output directory exists for file: {filepath} (Directory: {dir_path})")
        except OSError as e:
            logging.error(f"Could not create output directory {dir_path} for file {filepath}. {e}")
            raise 

def create_audio_manifest(input_dir, output_file_path):
    """
    Scans the input_dir for WAV files and generates a manifest file in JSONL format.
    Each line in the manifest file is a JSON object containing the audio file's absolute path
    and its duration in seconds.

    Args:
        input_dir (str): Directory containing the audio files (WAV format).
        output_file_path (str): Path for the output manifest file.
    
    Returns:
        bool: True if manifest generation was successful (or at least attempted), False on critical error (e.g. dir creation).
    """
    abs_input_dir = os.path.abspath(input_dir)
    abs_output_file = os.path.abspath(output_file_path)

    logging.info(f"--- Starting Manifest Generation ---")
    logging.info(f"Scanning input directory for WAV files: {abs_input_dir}")
    logging.info(f"Manifest will be written to: {abs_output_file}")

    if not os.path.isdir(abs_input_dir):
        logging.error(f"Input directory '{abs_input_dir}' not found or is not a directory.")
        return False

    try:
        ensure_output_dir_for_file(abs_output_file)
    except Exception:
        logging.error(f"Exiting manifest generation due to output directory creation failure for {abs_output_file}.")
        return False

    files_processed_count = 0
    files_error_count = 0
    
    # Ensure we're working with POSIX-style paths for consistency in manifests, if desired.
    # Though absolute paths are generally fine. Forcing POSIX might be an option if required by specific tools.
    # For now, os.path.abspath will use the OS-native format.

    try:
        with open(abs_output_file, 'w', encoding='utf-8') as f_out:
            for root, _, files in os.walk(abs_input_dir):
                for file in files:
                    if file.lower().endswith(".wav"):
                        filepath = os.path.join(root, file)
                        # Get absolute path consistently
                        abs_filepath = os.path.abspath(filepath) 
                        
                        try:
                            audio = AudioSegment.from_wav(abs_filepath)
                            duration_seconds = audio.duration_seconds
                            
                            manifest_entry = {
                                "audio_filepath": abs_filepath, # Storing absolute path
                                "duration": round(duration_seconds, 3) # Round duration to 3 decimal places
                            }
                            
                            f_out.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                            files_processed_count += 1
                            logging.debug(f"Added to manifest: {abs_filepath}, Duration: {duration_seconds:.3f}s")

                        except CouldntDecodeError:
                            logging.warning(f"Could not decode WAV file {abs_filepath}. File might be corrupted. Skipping.")
                            files_error_count += 1
                        except Exception as e:
                            logging.warning(f"Error processing file {abs_filepath}: {e}. Skipping.")
                            files_error_count += 1
                            
    except IOError as e:
        logging.error(f"Could not write to manifest file {abs_output_file}: {e}")
        return False
                        
    logging.info(f"Manifest generation complete.")
    logging.info(f"Successfully processed and added to manifest: {files_processed_count} files.")
    logging.info(f"Files skipped due to errors: {files_error_count}")
    return True


def main_cli():
    """CLI entry point for generating an audio manifest."""
    parser = argparse.ArgumentParser(description="Generate an audio manifest file (JSONL) from a directory of WAV files.")
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help="Directory containing the segmented audio files (WAV format)."
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help="Path for the output manifest file (e.g., data/manifest/audio_manifest.jsonl)."
    )

    args = parser.parse_args()
    create_audio_manifest(args.input_dir, args.output_file)

if __name__ == "__main__":
    main_cli()
