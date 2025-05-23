import argparse
import os
import subprocess

# Define the output directory for raw YouTube audio data
# Assuming the script is in runyoro_speech_ai/data_ingestion/
# So, ../data/raw/youtube/ will point to runyoro_speech_ai/data/raw/youtube/
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'youtube')

def download_audio(video_url, output_dir):
    """
    Downloads the audio track of a given YouTube URL using yt-dlp.

    Args:
        video_url (str): The URL of the YouTube video.
        output_dir (str): The directory where the audio will be saved.
    """
    print(f"Attempting to download audio for: {video_url}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # yt-dlp command:
    # -x: extract audio
    # --audio-format best: choose the best audio format (yt-dlp will pick one like m4a, opus, etc.)
    # -o: output template. Saves to output_dir/video_id.extension
    # --no-playlist: if a playlist URL is given, download only the video, not the whole playlist
    # --embed-metadata: embed metadata like title, artist if possible
    # --quiet: suppress yt-dlp console output unless errors
    # --progress: show progress bar (can be verbose, consider removing if running many downloads)
    # video_url: the URL to download from
    
    # Using %(id)s.%(ext)s ensures unique filenames based on video ID.
    output_template = os.path.join(output_dir, '%(id)s.%(ext)s')

    command = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'best',
        '-o', output_template,
        '--no-playlist', # Process only single video if URL is part of a playlist
        '--embed-metadata',
        '--quiet', # Be less verbose
        # '--progress', # Uncomment for progress bar, can be noisy
        video_url
    ]

    try:
        # Using subprocess.run for simplicity. For more complex scenarios (e.g., real-time output processing),
        # Popen might be more suitable.
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded audio for: {video_url}. Output at {output_dir}")
        # print(f"yt-dlp output: {result.stdout}") # Uncomment for debugging if needed
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {video_url}: yt-dlp failed.")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print("Error: yt-dlp command not found. Please ensure yt-dlp is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred while trying to download {video_url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download audio from YouTube videos.")
    
    # Option 1: Provide URLs directly on the command line
    parser.add_argument(
        '--urls', 
        nargs='+', 
        metavar='URL',
        help="One or more YouTube video URLs separated by spaces."
    )
    
    # Option 2: Provide a file containing URLs
    parser.add_argument(
        '--url-file', 
        type=str, 
        metavar='FILE_PATH',
        help="Path to a text file containing YouTube URLs (one URL per line)."
    )
    
    # Option to specify output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save downloaded audio. Defaults to: {DEFAULT_OUTPUT_DIR}"
    )

    args = parser.parse_args()

    video_urls_to_process = []

    if args.urls:
        video_urls_to_process.extend(args.urls)
    
    if args.url_file:
        try:
            with open(args.url_file, 'r') as f:
                file_urls = [line.strip() for line in f if line.strip()]
                video_urls_to_process.extend(file_urls)
        except FileNotFoundError:
            print(f"Error: URL file not found at {args.url_file}")
            return # Exit if the file is not found
        except Exception as e:
            print(f"Error reading URL file {args.url_file}: {e}")
            return

    if not video_urls_to_process:
        print("No URLs provided. Please specify URLs via --urls or --url-file.")
        parser.print_help()
        return

    # Ensure the specified output directory (or default) exists
    # This is also done in download_audio, but good to ensure it early if processing many files.
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory set to: {args.output_dir}")

    for url in video_urls_to_process:
        download_audio(url, args.output_dir)

if __name__ == "__main__":
    main()
