import argparse
import os
import logging
import sys

# Ensure the scripts directory is in the Python path if generate_manifest is there
# This assumes main_ingest.py is in data_ingestion, and scripts/ is a sibling to data_ingestion's parent.
# Adjust if your structure is different.
# For runyoro_speech_ai/data_ingestion/main_ingest.py
# and runyoro_speech_ai/scripts/generate_manifest.py
# Parent of data_ingestion is runyoro_speech_ai/
# So, scripts dir is runyoro_speech_ai/scripts/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
scripts_dir = os.path.join(project_root, 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import functions from other ingestion scripts
from download_youtube import orchestrate_youtube_downloads
from process_local_files import ingest_local_media
from preprocess_audio import run_conversion_stage, run_segmentation_stage
try:
    from generate_manifest import create_audio_manifest
except ImportError:
    logging.error("Could not import 'create_audio_manifest' from 'generate_manifest.py'. Ensure it's in the 'scripts' directory and accessible.")
    # Define a dummy function to prevent NameError if import fails, pipeline will fail later if this stage is run.
    def create_audio_manifest(*args, **kwargs):
        logging.error("generate_manifest.create_audio_manifest is not available due to import error.")
        return False


# --- Default Paths Configuration ---
# These paths are relative to the project root (runyoro_speech_ai/)
DEFAULT_BASE_DATA_DIR = os.path.join(project_root, 'data')

DEFAULT_YT_DOWNLOAD_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, 'raw', 'youtube_downloads')
DEFAULT_LOCAL_UPLOAD_SAMPLE_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, 'raw', 'local_uploads_sample') # Sample for user
DEFAULT_LOCAL_INGEST_TARGET_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, 'raw', 'local_ingested') # Target for copied local files

DEFAULT_CONVERSION_OUTPUT_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, 'processed', 'converted_audio_16khz_mono')
DEFAULT_SEGMENTATION_OUTPUT_DIR = os.path.join(DEFAULT_BASE_DATA_DIR, 'processed', 'segmented_audio')
DEFAULT_MANIFEST_FILE = os.path.join(DEFAULT_BASE_DATA_DIR, 'manifest', 'audio_manifest.jsonl')


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
            # You can add logging.FileHandler here if you want to log to a file
        ]
    )
    # Silence pydub's own logger if it's too verbose, or set its level
    # logging.getLogger("pydub").setLevel(logging.WARNING)


def main_orchestrator():
    setup_logging()
    logger = logging.getLogger(__name__) # Get a logger for this module

    parser = argparse.ArgumentParser(description="Runyoro Speech AI Data Ingestion Pipeline.")

    # --- Stage Control Flags ---
    parser.add_argument('--skip-youtube-download', action='store_true', help="Skip YouTube video audio download stage.")
    parser.add_argument('--skip-local-ingest', action='store_true', help="Skip local media file ingestion stage.")
    parser.add_argument('--skip-conversion', action='store_true', help="Skip audio conversion to 16kHz mono WAV.")
    parser.add_argument('--skip-segmentation', action='store_true', help="Skip audio segmentation stage.")
    parser.add_argument('--skip-manifest', action='store_true', help="Skip manifest generation stage.")

    # --- YouTube Download Arguments ---
    yt_group = parser.add_argument_group('YouTube Download Options')
    yt_group.add_argument('--yt-urls', nargs='+', metavar='URL', help="List of YouTube video URLs.")
    yt_group.add_argument('--yt-url-file', type=str, metavar='FILE', help="Path to a file containing YouTube URLs (one per line).")
    yt_group.add_argument('--yt-output-dir', type=str, default=DEFAULT_YT_DOWNLOAD_DIR, help=f"Directory to save downloaded YouTube audio. Default: {DEFAULT_YT_DOWNLOAD_DIR}")

    # --- Local File Ingestion Arguments ---
    local_group = parser.add_argument_group('Local File Ingestion Options')
    local_group.add_argument('--local-upload-dir', type=str, default=DEFAULT_LOCAL_UPLOAD_SAMPLE_DIR, help=f"Directory containing user-uploaded local media files. Default: {DEFAULT_LOCAL_UPLOAD_SAMPLE_DIR}")
    local_group.add_argument('--local-ingest-target-dir', type=str, default=DEFAULT_LOCAL_INGEST_TARGET_DIR, help=f"Directory to copy raw local media files to. Default: {DEFAULT_LOCAL_INGEST_TARGET_DIR}")

    # --- Audio Conversion Arguments ---
    conv_group = parser.add_argument_group('Audio Conversion Options (to 16kHz mono WAV)')
    conv_group.add_argument('--conversion-input-dir', type=str, help="Directory of raw media for conversion. If not set, uses outputs from YouTube & Local Ingest stages if they ran.")
    conv_group.add_argument('--conversion-output-dir', type=str, default=DEFAULT_CONVERSION_OUTPUT_DIR, help=f"Directory to save converted 16kHz mono WAV files. Default: {DEFAULT_CONVERSION_OUTPUT_DIR}")

    # --- Audio Segmentation Arguments ---
    seg_group = parser.add_argument_group('Audio Segmentation Options')
    seg_group.add_argument('--segmentation-input-dir', type=str, help=f"Directory of 16kHz mono WAVs for segmentation. If not set, uses --conversion-output-dir. Default: (output of conversion stage)")
    seg_group.add_argument('--segmentation-output-dir', type=str, default=DEFAULT_SEGMENTATION_OUTPUT_DIR, help=f"Directory to save segmented audio clips. Default: {DEFAULT_SEGMENTATION_OUTPUT_DIR}")
    # Parameters for preprocess_audio.py's segmentation
    seg_group.add_argument('--min-silence-len', type=int, default=700, help="Min silence length (ms) for VAD. Default: 700.")
    seg_group.add_argument('--silence-thresh', type=int, default=-45, help="Silence threshold (dBFS) for VAD. Default: -45.")
    seg_group.add_argument('--keep-silence', type=int, default=250, help="Keep silence (ms) at segment ends. Default: 250.")
    seg_group.add_argument('--min-segment-duration', type=int, default=2000, help="Min segment duration (ms). Default: 2000 (2s).")
    seg_group.add_argument('--max-segment-duration', type=int, default=20000, help="Max segment duration (ms) before force split. Default: 20000 (20s).")
    seg_group.add_argument('--target-split-duration', type=int, default=10000, help="Target duration (ms) for force-split segments. Default: 10000 (10s).")

    # --- Manifest Generation Arguments ---
    manifest_group = parser.add_argument_group('Manifest Generation Options')
    manifest_group.add_argument('--manifest-input-dir', type=str, help=f"Directory of final segmented audio clips for manifest. If not set, uses --segmentation-output-dir. Default: (output of segmentation stage)")
    manifest_group.add_argument('--manifest-output-file', type=str, default=DEFAULT_MANIFEST_FILE, help=f"Path to save the generated manifest JSONL file. Default: {DEFAULT_MANIFEST_FILE}")

    args = parser.parse_args()
    
    logger.info("Starting Runyoro Speech AI Data Ingestion Pipeline...")
    logger.info(f"Pipeline configuration: {vars(args)}")

    # --- Determine effective paths for each stage ---
    # These will hold the actual output paths of each stage, to be used as input for the next.
    effective_yt_output_dir = os.path.abspath(args.yt_output_dir)
    effective_local_ingest_target_dir = os.path.abspath(args.local_ingest_target_dir)
    effective_conversion_output_dir = os.path.abspath(args.conversion_output_dir)
    effective_segmentation_output_dir = os.path.abspath(args.segmentation_output_dir)
    effective_manifest_output_file = os.path.abspath(args.manifest_output_file)

    # --- Stage 1: YouTube Download ---
    if not args.skip_youtube_download:
        logger.info("--- Stage: YouTube Audio Download ---")
        if not args.yt_urls and not args.yt_url_file:
            logger.warning("YouTube download stage selected, but no URLs or URL file provided. Skipping.")
        else:
            if not orchestrate_youtube_downloads(urls_list=args.yt_urls, url_file_path=args.yt_url_file, output_dir=effective_yt_output_dir):
                logger.error("YouTube download stage encountered errors.")
                # Decide if pipeline should stop on error. For now, continue.
    else:
        logger.info("Skipping YouTube Audio Download stage.")

    # --- Stage 2: Local File Ingestion ---
    if not args.skip_local_ingest:
        logger.info("--- Stage: Local Media Ingestion ---")
        abs_local_upload_dir = os.path.abspath(args.local_upload_dir)
        if not os.path.isdir(abs_local_upload_dir) or not os.listdir(abs_local_upload_dir): # Check if dir exists and is not empty
             logger.warning(f"Local upload directory {abs_local_upload_dir} is empty or not found. Skipping local media ingestion.")
        else:
            if not ingest_local_media(user_upload_dir=abs_local_upload_dir, raw_files_target_dir=effective_local_ingest_target_dir):
                logger.error("Local media ingestion stage encountered errors.")
    else:
        logger.info("Skipping Local Media Ingestion stage.")

    # --- Stage 3: Audio Conversion (to 16kHz mono WAV) ---
    conversion_inputs = []
    if args.conversion_input_dir: # User explicitly provided a directory for conversion
        conversion_inputs.append(os.path.abspath(args.conversion_input_dir))
    else: # Auto-detect inputs from previous stages if they ran
        if not args.skip_youtube_download and os.path.isdir(effective_yt_output_dir) and os.listdir(effective_yt_output_dir):
            conversion_inputs.append(effective_yt_output_dir)
        if not args.skip_local_ingest and os.path.isdir(effective_local_ingest_target_dir) and os.listdir(effective_local_ingest_target_dir):
            conversion_inputs.append(effective_local_ingest_target_dir)
            
    if not args.skip_conversion:
        logger.info("--- Stage: Audio Conversion to 16kHz mono WAV ---")
        if not conversion_inputs:
            logger.warning("Audio conversion stage selected, but no input directories found (from YouTube, local ingest, or explicit --conversion-input-dir). Skipping.")
        else:
            for i, input_path in enumerate(conversion_inputs):
                logger.info(f"Running conversion for input directory {i+1}/{len(conversion_inputs)}: {input_path}")
                if not os.path.isdir(input_path):
                    logger.warning(f"  Skipping conversion for {input_path} as it's not a valid directory or is empty.")
                    continue
                if not run_conversion_stage(input_dir=input_path, conversion_output_dir=effective_conversion_output_dir):
                    logger.warning(f"  Conversion from {input_path} attempted but might have had issues or no applicable files.")
            logger.info(f"All conversion attempts finished. Output in: {effective_conversion_output_dir}")
    else:
        logger.info("Skipping Audio Conversion stage.")

    # --- Stage 4: Audio Segmentation ---
    segmentation_input_actual = args.segmentation_input_dir if args.segmentation_input_dir else effective_conversion_output_dir
    abs_segmentation_input_actual = os.path.abspath(segmentation_input_actual)

    if not args.skip_segmentation:
        logger.info("--- Stage: Audio Segmentation ---")
        if not os.path.isdir(abs_segmentation_input_actual) or not os.listdir(abs_segmentation_input_actual):
            logger.warning(f"Segmentation stage selected, but input directory {abs_segmentation_input_actual} is empty or not found. Skipping.")
        else:
            if not run_segmentation_stage(
                segmentation_input_dir=abs_segmentation_input_actual,
                segmentation_output_dir=effective_segmentation_output_dir,
                min_silence_len=args.min_silence_len,
                silence_thresh=args.silence_thresh,
                keep_silence=args.keep_silence,
                min_duration_ms=args.min_segment_duration,
                max_duration_ms=args.max_segment_duration,
                target_split_duration_ms=args.target_split_duration
            ):
                logger.warning("Segmentation stage attempted but might have had issues or no applicable files.")
            logger.info(f"Segmentation finished. Output in: {effective_segmentation_output_dir}")
    else:
        logger.info("Skipping Audio Segmentation stage.")

    # --- Stage 5: Manifest Generation ---
    manifest_input_actual = args.manifest_input_dir if args.manifest_input_dir else effective_segmentation_output_dir
    abs_manifest_input_actual = os.path.abspath(manifest_input_actual)

    if not args.skip_manifest:
        logger.info("--- Stage: Manifest Generation ---")
        if not os.path.isdir(abs_manifest_input_actual) or not os.listdir(abs_manifest_input_actual):
            logger.warning(f"Manifest generation stage selected, but input directory {abs_manifest_input_actual} is empty or not found. Skipping.")
        else:
            if 'create_audio_manifest' not in globals() or not callable(create_audio_manifest):
                 logger.error("Manifest generation function is not available. Skipping.")
            elif not create_audio_manifest(input_dir=abs_manifest_input_actual, output_file_path=effective_manifest_output_file):
                logger.error("Manifest generation stage encountered errors.")
            else:
                logger.info(f"Manifest generation finished. Output file: {effective_manifest_output_file}")
    else:
        logger.info("Skipping Manifest Generation stage.")

    logger.info("Runyoro Speech AI Data Ingestion Pipeline finished.")


if __name__ == "__main__":
    main_orchestrator()
