import argparse
import os
import json
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

def ensure_output_dir(filepath):
    """Creates the directory for the given filepath if it doesn't exist."""
    dir_path = os.path.dirname(filepath)
    if dir_path: # Ensure dir_path is not empty (e.g. if output is in current dir)
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Ensured output directory exists: {dir_path}")
        except OSError as e:
            print(f"Error: Could not create output directory {dir_path}. {e}")
            raise # Re-raise to stop execution if output dir can't be made

def generate_manifest(input_dir, output_file):
    """
    Scans the input directory for WAV files and generates a manifest file in JSONL format.
    Each line in the manifest file is a JSON object containing the audio file's absolute path
    and its duration in seconds.
    """
    print(f"Scanning input directory: {input_dir}")
    print(f"Manifest will be written to: {output_file}")

    try:
        ensure_output_dir(output_file)
    except Exception:
        print(f"Exiting due to output directory creation failure for manifest file.")
        return

    files_processed = 0
    files_error = 0

    with open(output_file, 'w') as f_out:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".wav"):
                    filepath = os.path.join(root, file)
                    abs_filepath = os.path.abspath(filepath)
                    
                    try:
                        # Get audio duration using pydub
                        audio = AudioSegment.from_wav(abs_filepath)
                        duration_seconds = audio.duration_seconds
                        
                        manifest_entry = {
                            "audio_filepath": abs_filepath,
                            "duration": duration_seconds
                        }
                        
                        # Write the JSON object as a line in the output file
                        f_out.write(json.dumps(manifest_entry) + '\n')
                        files_processed += 1
                        # print(f"Processed: {abs_filepath}, Duration: {duration_seconds:.2f}s")

                    except CouldntDecodeError:
                        print(f"Error: Could not decode WAV file {abs_filepath}. File might be corrupted.")
                        files_error += 1
                    except Exception as e:
                        print(f"Error processing file {abs_filepath}: {e}")
                        files_error += 1
                        
    print(f"\nManifest generation complete.")
    print(f"Successfully processed and added to manifest: {files_processed} files.")
    print(f"Files skipped due to errors: {files_error}")

def main():
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

    abs_input_dir = os.path.abspath(args.input_dir)
    abs_output_file = os.path.abspath(args.output_file)

    # Validate input directory
    if not os.path.isdir(abs_input_dir):
        print(f"Error: Input directory '{abs_input_dir}' not found or is not a directory.")
        return
        
    generate_manifest(abs_input_dir, abs_output_file)

if __name__ == "__main__":
    main()
