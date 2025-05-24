import argparse
import os
import shutil
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the target base directory for processed local files
# Assuming the script is in runyoro_speech_ai/data_ingestion/
# So, ../data/raw/local_source/ will point to runyoro_speech_ai/data/raw/local_source/
DEFAULT_TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'local_source'))

# Define supported audio and video extensions
SUPPORTED_EXTENSIONS = [
    '.wav', '.mp3', '.m4a', '.ogg', '.flac',  # Audio
    '.mp4', '.mkv', '.mov', '.avi', '.webm'   # Video
]

def ingest_local_media(user_upload_dir, raw_files_target_dir):
    """
    Scans the user_upload_dir for supported media files and copies them to raw_files_target_dir.

    Args:
        user_upload_dir (str): The directory to scan for media files.
        raw_files_target_dir (str): The directory where recognized media files will be copied.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    abs_input_dir = os.path.abspath(user_upload_dir)
    abs_target_dir = os.path.abspath(raw_files_target_dir)

    logging.info(f"Scanning user upload directory: {abs_input_dir}")
    logging.info(f"Target directory for raw local files: {abs_target_dir}")

    if not os.path.isdir(abs_input_dir):
        logging.error(f"Error: Input directory '{abs_input_dir}' not found or is not a directory.")
        return False

    try:
        os.makedirs(abs_target_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Error: Could not create target directory {abs_target_dir}. {e}")
        return False

    files_copied_count = 0
    files_skipped_count = 0
    error_copying_count = 0

    for item in os.listdir(abs_input_dir):
        item_path = os.path.join(abs_input_dir, item)
        
        if os.path.isfile(item_path):
            _, ext = os.path.splitext(item)
            if ext.lower() in SUPPORTED_EXTENSIONS:
                target_file_path = os.path.join(abs_target_dir, item)
                try:
                    if os.path.exists(target_file_path):
                        # logging.info(f"File {item} already exists in target directory. Skipping copy.")
                        files_skipped_count +=1
                        continue

                    shutil.copy2(item_path, target_file_path) # copy2 preserves metadata
                    logging.info(f"Copied: {item} to {target_file_path}")
                    files_copied_count += 1
                except shutil.Error as e:
                    logging.error(f"Error copying file {item} using shutil: {e}")
                    error_copying_count +=1
                except IOError as e:
                    logging.error(f"Error copying file {item} (IOError): {e}")
                    error_copying_count +=1
            else:
                # logging.debug(f"Skipped (unsupported extension): {item}")
                files_skipped_count +=1
        else:
            # logging.debug(f"Skipped (not a file): {item}")
            files_skipped_count +=1
            
    logging.info(f"Local media ingestion complete.")
    logging.info(f"Files copied: {files_copied_count}")
    logging.info(f"Files skipped (unsupported or already exist): {files_skipped_count}")
    logging.info(f"Files failed to copy: {error_copying_count}")
    
    return error_copying_count == 0


def main_cli():
    """CLI entry point for processing local files."""
    parser = argparse.ArgumentParser(description="Process local audio/video files and copy them to a target directory for raw ingestion.")
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help="Path to the directory where users have dropped their local audio/video files."
    )
    
    parser.add_argument(
        '--target-dir',
        type=str,
        default=DEFAULT_TARGET_DIR,
        help=f"Directory to save ingested raw media files. Defaults to: {DEFAULT_TARGET_DIR}"
    )

    args = parser.parse_args()

    ingest_local_media(args.input_dir, args.target_dir)

if __name__ == "__main__":
    main_cli()
