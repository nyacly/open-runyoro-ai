import argparse
import os
import subprocess
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the output directory for raw YouTube audio data
# Assuming the script is in runyoro_speech_ai/data_ingestion/
# So, ../data/raw/youtube/ will point to runyoro_speech_ai/data/raw/youtube/
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'youtube'))

def download_single_video(video_url, output_dir):
    """
    Downloads the audio track of a given YouTube URL using yt-dlp.

    Args:
        video_url (str): The URL of the YouTube video.
        output_dir (str): The directory where the audio will be saved.
    Returns:
        bool: True if download was successful or file already exists, False otherwise.
    """
    logging.info(f"Attempting to download audio for: {video_url} into {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Using %(id)s.%(ext)s ensures unique filenames based on video ID.
    # We expect yt-dlp to handle if the specific audio format is already downloaded.
    output_template = os.path.join(output_dir, '%(id)s.%(ext)s')

    # Check if a file with this ID already exists (any extension)
    # This is a simple check; yt-dlp might have more robust ways to handle already downloaded files.
    # For now, if we find *a* file that matches `%(id)s.*`, we assume it's downloaded.
    # A more robust check would involve yt-dlp's --download-archive feature or parsing its output.
    # This check is basic:
    # video_id_for_check = video_url.split("v=")[-1].split("&")[0] # Basic ID extraction
    # if any(f.startswith(video_id_for_check) for f in os.listdir(output_dir)):
    #     logging.info(f"A file possibly related to video ID {video_id_for_check} already exists in {output_dir}. Skipping download.")
    #     return True 
    # Note: The above check is too simplistic and can be error-prone. yt-dlp itself handles already downloaded files well if the exact output path matches.
    # We'll rely on yt-dlp's implicit check or allow it to overwrite/create new if format differs.

    command = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'best', # yt-dlp will pick a suitable audio format like m4a, opus, etc.
        '-o', output_template,
        '--no-playlist', 
        '--embed-metadata',
        '--quiet',
        # '--progress', # Can be noisy for many files
        video_url
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info(f"Successfully downloaded audio for: {video_url}. Output in {output_dir}")
        # logging.debug(f"yt-dlp output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading {video_url}: yt-dlp failed.")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return code: {e.returncode}")
        # yt-dlp often outputs useful error messages to stderr
        logging.error(f"Stderr: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        logging.error("Error: yt-dlp command not found. Please ensure yt-dlp is installed and in your PATH.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while trying to download {video_url}: {e}")
        return False

def orchestrate_youtube_downloads(urls_list=None, url_file_path=None, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Orchestrates the download of audio from a list of YouTube URLs or a file containing URLs.

    Args:
        urls_list (list, optional): A list of YouTube video URLs.
        url_file_path (str, optional): Path to a text file with one URL per line.
        output_dir (str): Directory to save downloaded audio. Defaults to DEFAULT_OUTPUT_DIR.

    Returns:
        bool: True if all downloads attempted were successful (or already existed), False otherwise.
    """
    video_urls_to_process = []
    if urls_list:
        video_urls_to_process.extend(urls_list)
    
    if url_file_path:
        abs_url_file_path = os.path.abspath(url_file_path)
        logging.info(f"Reading URLs from file: {abs_url_file_path}")
        try:
            with open(abs_url_file_path, 'r') as f:
                file_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                video_urls_to_process.extend(file_urls)
                logging.info(f"Found {len(file_urls)} URLs in {abs_url_file_path}.")
        except FileNotFoundError:
            logging.error(f"Error: URL file not found at {abs_url_file_path}")
            return False
        except Exception as e:
            logging.error(f"Error reading URL file {abs_url_file_path}: {e}")
            return False

    if not video_urls_to_process:
        logging.warning("No URLs provided to download.")
        return True # No action needed, so considered "successful"

    # Ensure the specified output directory (or default) exists
    abs_output_dir = os.path.abspath(output_dir)
    try:
        os.makedirs(abs_output_dir, exist_ok=True)
        logging.info(f"YouTube download output directory set to: {abs_output_dir}")
    except OSError as e:
        logging.error(f"Could not create or access output directory {abs_output_dir}: {e}")
        return False

    overall_success = True
    for i, url in enumerate(video_urls_to_process):
        logging.info(f"Processing URL {i+1}/{len(video_urls_to_process)}: {url}")
        if not download_single_video(url, abs_output_dir):
            overall_success = False # Mark failure if any download fails
            
    if overall_success:
        logging.info("All YouTube downloads attempted successfully (or files already existed).")
    else:
        logging.warning("Some YouTube downloads failed. Check logs above.")
        
    return overall_success

def main_cli():
    """CLI entry point for downloading YouTube audio."""
    parser = argparse.ArgumentParser(description="Download audio from YouTube videos.")
    
    parser.add_argument(
        '--urls', 
        nargs='+', 
        metavar='URL',
        help="One or more YouTube video URLs separated by spaces."
    )
    parser.add_argument(
        '--url-file', 
        type=str, 
        metavar='FILE_PATH',
        help="Path to a text file containing YouTube URLs (one URL per line)."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save downloaded audio. Defaults to: {DEFAULT_OUTPUT_DIR}"
    )

    args = parser.parse_args()

    if not args.urls and not args.url_file:
        parser.error("No input specified. Please provide URLs via --urls or --url-file.")

    orchestrate_youtube_downloads(
        urls_list=args.urls, 
        url_file_path=args.url_file, 
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main_cli()
