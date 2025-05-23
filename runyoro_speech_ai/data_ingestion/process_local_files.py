import argparse
import os
import shutil

# Define the target base directory for processed local files
# Assuming the script is in runyoro_speech_ai/data_ingestion/
# So, ../data/raw/local_source/ will point to runyoro_speech_ai/data/raw/local_source/
DEFAULT_TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'local_source'))

# Define supported audio and video extensions
SUPPORTED_EXTENSIONS = [
    '.wav', '.mp3', '.m4a', '.ogg', '.flac',  # Audio
    '.mp4', '.mkv', '.mov', '.avi', '.webm'   # Video
]

def copy_media_files(input_dir, target_dir):
    """
    Scans the input directory for supported media files and copies them to the target directory.

    Args:
        input_dir (str): The directory to scan for media files.
        target_dir (str): The directory where recognized media files will be copied.
    """
    print(f"Scanning input directory: {input_dir}")
    print(f"Target directory for copied files: {target_dir}")

    # Ensure the target directory exists
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create target directory {target_dir}. {e}")
        return

    files_copied_count = 0
    files_skipped_count = 0

    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        
        if os.path.isfile(item_path):
            _, ext = os.path.splitext(item)
            if ext.lower() in SUPPORTED_EXTENSIONS:
                target_file_path = os.path.join(target_dir, item)
                try:
                    # For now, simple copy. Consider unique naming if clashes are a concern.
                    # e.g., target_file_path = os.path.join(target_dir, f"{os.path.splitext(item)[0]}_{int(time.time())}{ext}")
                    if os.path.exists(target_file_path):
                        print(f"Info: File {item} already exists in target directory. Skipping copy.")
                        files_skipped_count +=1
                        continue

                    shutil.copy2(item_path, target_file_path) # copy2 preserves metadata
                    print(f"Copied: {item} to {target_file_path}")
                    files_copied_count += 1
                except shutil.Error as e:
                    print(f"Error copying file {item}: {e}")
                except IOError as e:
                    print(f"Error copying file {item} (IOError): {e}")
            else:
                print(f"Skipped (unsupported extension): {item}")
                files_skipped_count +=1
        else:
            print(f"Skipped (not a file): {item}")
            files_skipped_count +=1
            
    print(f"\nProcessing complete.")
    print(f"Files copied: {files_copied_count}")
    print(f"Files skipped: {files_skipped_count}")


def main():
    parser = argparse.ArgumentParser(description="Process local audio/video files and copy them to a target directory.")
    
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
        help=f"Directory to save processed media files. Defaults to: {DEFAULT_TARGET_DIR}"
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found or is not a directory.")
        return

    copy_media_files(args.input_dir, args.target_dir)

if __name__ == "__main__":
    main()
