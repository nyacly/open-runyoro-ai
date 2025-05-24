import argparse
import os
import shutil
import logging
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.silence import split_on_silence

# Configure basic logging - This will be configured by the main_ingest.py if imported.
# If run standalone, this basic config will apply.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Define supported audio and video extensions for initial conversion processing
SUPPORTED_CONVERSION_EXTENSIONS = [
    '.wav', '.mp3', '.m4a', '.ogg', '.flac',  # Audio
    '.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv'   # Video
]

# Target sampling rate (16kHz is common for ASR)
TARGET_SAMPLING_RATE = 16000

def ensure_output_dir(directory_path):
    """Creates the directory if it doesn't exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        logging.debug(f"Ensured output directory exists: {directory_path}")
    except OSError as e:
        logging.error(f"Could not create output directory {directory_path}. {e}")
        raise

def convert_to_standard_wav(filepath, conversion_output_dir):
    """
    Processes a single media file:
    - Extracts audio if video.
    - Converts to mono.
    - Resamples to TARGET_SAMPLING_RATE (16kHz).
    - Saves as WAV in the conversion_output_dir.
    Returns the path to the converted WAV file or None if failed.
    """
    filename = os.path.basename(filepath)
    basename, _ = os.path.splitext(filename)
    
    output_filename = f"{basename}.wav" # Standardized output name
    output_filepath = os.path.join(conversion_output_dir, output_filename)

    if os.path.exists(output_filepath):
        logging.info(f"Converted file {output_filepath} already exists. Skipping conversion.")
        return output_filepath # Return existing path

    try:
        logging.info(f"Converting: {filepath}")
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_channels(1)
        logging.debug(f"  Converted to mono for {filepath}")
        audio = audio.set_frame_rate(TARGET_SAMPLING_RATE)
        logging.debug(f"  Resampled to {TARGET_SAMPLING_RATE}Hz for {filepath}")
        audio.export(output_filepath, format="wav")
        logging.info(f"  Successfully converted and saved to {output_filepath}")
        return output_filepath
    except CouldntDecodeError:
        logging.error(f"Could not decode {filepath}. File might be corrupted or an unsupported format.")
    except FileNotFoundError: # This error is typically for ffmpeg/ffprobe not being found
        logging.error(f"ffmpeg/ffprobe not found for pydub processing {filepath}. Please ensure it's installed and in PATH.")
    except Exception as e:
        logging.error(f"Error converting file {filepath}: {e}")
    return None

def run_conversion_stage(input_dir, conversion_output_dir):
    """
    Recursively scans input_dir for media files and processes them using convert_to_standard_wav.
    Returns True if any processing was attempted (success or fail), False if no applicable files found.
    """
    logging.info(f"--- Starting Audio Conversion Stage ---")
    logging.info(f"Scanning for media files in: {input_dir}")
    logging.info(f"Converted 16kHz mono WAVs will be saved to: {conversion_output_dir}")
    
    ensure_output_dir(conversion_output_dir) # Handles exception if dir creation fails
    converted_files_count = 0
    error_files_count = 0
    attempted_processing = False

    for root, _, files in os.walk(input_dir):
        logging.debug(f"Scanning directory: {root}")
        if not files:
            logging.debug(f"No files found in directory: {root}")
        for file in files:
            logging.debug(f"Found file: {file} in {root}")
            filepath = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            logging.debug(f"  File extension: {ext} (lowercase: {ext.lower()})")
            
            logging.debug(f"  Checking if '{ext.lower()}' is in {SUPPORTED_CONVERSION_EXTENSIONS}")
            if ext.lower() in SUPPORTED_CONVERSION_EXTENSIONS:
                attempted_processing = True
                if convert_to_standard_wav(filepath, conversion_output_dir):
                    converted_files_count += 1
                else:
                    error_files_count += 1
            else:
                logging.debug(f"  Skipping file {file}: extension '{ext.lower()}' not in supported list.")
    
    logging.info(f"Conversion Stage Complete.")
    logging.info(f"Successfully converted files: {converted_files_count}")
    logging.info(f"Files failed to convert: {error_files_count}")
    return attempted_processing


def force_split_segment(audio_segment, target_duration_ms, min_split_duration_ms, base_output_path, original_filename_prefix):
    """
    Splits a long audio segment into smaller chunks of target_duration_ms.
    Returns the number of segments created.
    """
    num_chunks_created = 0
    duration = len(audio_segment)
    start_time = 0
    segment_counter = 0
    
    while start_time < duration:
        end_time = start_time + target_duration_ms
        current_chunk_duration = end_time - start_time
        
        if end_time > duration: # Last chunk
            current_chunk_duration = duration - start_time
            end_time = duration

        if current_chunk_duration < min_split_duration_ms:
            if segment_counter > 0: # If not the first chunk (i.e. original segment was split at least once)
                logging.info(f"    Force split: Last chunk too small ({current_chunk_duration}ms), discarding for {original_filename_prefix}.")
                break 
            # If it's the only chunk (original segment was < target_duration but > max_duration somehow, or just one very small chunk)
            # and it's less than min_split_duration, it will be filtered by min_duration_ms later if this is the only output.
            # For now, we let it pass here, assuming min_duration_ms check is done elsewhere for overall segment validity.
            
        chunk = audio_segment[start_time:end_time]
        segment_filename = f"{original_filename_prefix}_forced_split_{segment_counter:03d}.wav"
        chunk_output_path = os.path.join(base_output_path, segment_filename)
        
        try:
            chunk.export(chunk_output_path, format="wav")
            logging.info(f"    Force split: Saved chunk {segment_filename} ({len(chunk)}ms)")
            num_chunks_created += 1
        except Exception as e:
            logging.error(f"    Force split: Error exporting chunk {segment_filename}: {e}")
            
        segment_counter += 1
        start_time = end_time
        if start_time >= duration:
            break
            
    return num_chunks_created


def segment_wav_file(wav_filepath, segmentation_output_dir, min_silence_len, silence_thresh, 
                     keep_silence, min_duration_ms, max_duration_ms, target_split_duration_ms):
    """
    Segments a single WAV file based on silence and duration constraints.
    Returns number of segments successfully saved.
    """
    original_filename_no_ext = os.path.splitext(os.path.basename(wav_filepath))[0]
    logging.info(f"Segmenting: {wav_filepath}")

    try:
        audio = AudioSegment.from_wav(wav_filepath)
    except Exception as e:
        logging.error(f"  Error loading WAV file {wav_filepath}: {e}")
        return 0

    segments = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
        seek_step=1 
    )

    total_segments_saved = 0
    if not segments:
        logging.info(f"  No segments found for {wav_filepath} based on silence detection.")
        audio_len = len(audio)
        if audio_len > max_duration_ms:
             logging.info(f"  Original audio duration ({audio_len}ms) > max_duration_ms ({max_duration_ms}ms). Attempting force split.")
             total_segments_saved += force_split_segment(audio, target_split_duration_ms, min_duration_ms, segmentation_output_dir, original_filename_no_ext)
        elif audio_len >= min_duration_ms:
             segment_filename = f"{original_filename_no_ext}_full.wav"
             output_path = os.path.join(segmentation_output_dir, segment_filename)
             try:
                audio.export(output_path, format="wav")
                logging.info(f"  Saved full audio as one segment: {segment_filename} ({audio_len}ms)")
                total_segments_saved += 1
             except Exception as e:
                logging.error(f"  Error exporting full audio segment {segment_filename}: {e}")
        else:
            logging.info(f"  Full audio duration ({audio_len}ms) < min_duration_ms ({min_duration_ms}ms). Discarding.")
        return total_segments_saved


    segment_export_counter = 0 # Counter for successfully exported segments from this file
    for i, segment in enumerate(segments):
        seg_len = len(segment)
        logging.debug(f"  Segment {i}: duration {seg_len}ms from {original_filename_no_ext}")

        if seg_len < min_duration_ms:
            logging.info(f"    Discarded (too short): {seg_len}ms < {min_duration_ms}ms for segment {i} of {original_filename_no_ext}")
            continue
        
        if seg_len > max_duration_ms:
            logging.info(f"    Segment {i} too long ({seg_len}ms > {max_duration_ms}ms). Attempting force split for {original_filename_no_ext}")
            num_forced_chunks = force_split_segment(segment, target_split_duration_ms, min_duration_ms,
                                                    segmentation_output_dir, 
                                                    f"{original_filename_no_ext}_segment_{i:03d}")
            total_segments_saved += num_forced_chunks
        else:
            segment_filename = f"{original_filename_no_ext}_segment_{segment_export_counter:03d}.wav"
            output_path = os.path.join(segmentation_output_dir, segment_filename)
            try:
                segment.export(output_path, format="wav")
                logging.info(f"    Saved: {segment_filename} ({seg_len}ms)")
                total_segments_saved += 1
                segment_export_counter += 1
            except Exception as e:
                logging.error(f"    Error exporting segment {segment_filename}: {e}")
        
    logging.info(f"  Finished segmenting {wav_filepath}. Segments saved from this file: {total_segments_saved}")
    return total_segments_saved

def run_segmentation_stage(segmentation_input_dir, segmentation_output_dir, 
                           min_silence_len, silence_thresh, keep_silence, 
                           min_duration_ms, max_duration_ms, target_split_duration_ms):
    """
    Scans segmentation_input_dir for WAV files and segments them.
    Returns True if any processing was attempted, False otherwise.
    """
    logging.info(f"--- Starting Audio Segmentation Stage ---")
    logging.info(f"Scanning for 16kHz mono WAV files in: {segmentation_input_dir}")
    logging.info(f"Segmented audio clips will be saved to: {segmentation_output_dir}")

    ensure_output_dir(segmentation_output_dir)
    total_segments_generated_overall = 0
    files_processed_count = 0
    files_with_no_segments = 0
    attempted_processing = False

    for item in os.listdir(segmentation_input_dir):
        if item.lower().endswith(".wav"):
            attempted_processing = True
            filepath = os.path.join(segmentation_input_dir, item)
            if os.path.isfile(filepath): # Make sure it's a file
                files_processed_count +=1
                segments_from_file = segment_wav_file(
                    filepath, segmentation_output_dir,
                    min_silence_len, silence_thresh, keep_silence,
                    min_duration_ms, max_duration_ms, target_split_duration_ms
                )
                if segments_from_file > 0:
                    total_segments_generated_overall += segments_from_file
                else:
                    files_with_no_segments +=1
    
    logging.info(f"Segmentation Stage Complete.")
    logging.info(f"Processed {files_processed_count} WAV files.")
    logging.info(f"Total segments generated: {total_segments_generated_overall}")
    logging.info(f"Files resulting in no segments (after filtering): {files_with_no_segments}")
    return attempted_processing


def main_cli():
    """CLI entry point for preprocessing and segmenting audio."""
    parser = argparse.ArgumentParser(description="Preprocess (convert to 16kHz mono WAV) and segment audio files for ASR.")
    
    parser.add_argument('--input-dir', type=str, help="Path for raw audio/video files for conversion. Required if not skipping conversion.")
    parser.add_argument('--conversion-output-dir', type=str, help="Path to save converted 16kHz mono WAVs. Required if not skipping conversion.")
    
    parser.add_argument('--segmentation-input-dir', type=str, help="Path for 16kHz mono WAVs for segmentation. If not given, conversion-output-dir is used. Required if not skipping segmentation.")
    parser.add_argument('--segmentation-output-dir', type=str, help="Path to save segmented audio clips. Required if not skipping segmentation.")
    
    parser.add_argument('--skip-conversion', action='store_true', help="Skip the initial conversion stage.")
    parser.add_argument('--skip-segmentation', action='store_true', help="Skip audio segmentation.")

    parser.add_argument('--min-silence-len', type=int, default=700, help="Min silence length (ms). Default: 700.")
    parser.add_argument('--silence-thresh', type=int, default=-45, help="Silence threshold (dBFS). Default: -45.")
    parser.add_argument('--keep-silence', type=int, default=250, help="Keep silence (ms) at ends. Default: 250.")
    parser.add_argument('--min-segment-duration', type=int, default=2000, help="Min segment duration (ms). Default: 2000 (2s).")
    parser.add_argument('--max-segment-duration', type=int, default=20000, help="Max segment duration (ms) before force split. Default: 20000 (20s).")
    parser.add_argument('--target-split-duration', type=int, default=10000, help="Target duration (ms) for force-split segments. Default: 10000 (10s).")

    args = parser.parse_args()
    
    # Basic path validation for CLI mode
    processed_conversion = False
    if not args.skip_conversion:
        if not args.input_dir or not args.conversion_output_dir:
            parser.error("--input-dir and --conversion-output-dir are required if --skip-conversion is not set.")
        
        abs_input_dir = os.path.abspath(args.input_dir)
        abs_conversion_output_dir = os.path.abspath(args.conversion_output_dir)

        if not os.path.isdir(abs_input_dir):
            logging.error(f"Input directory '{abs_input_dir}' not found.")
            return
        
        try:
            ensure_output_dir(abs_conversion_output_dir)
        except Exception: return # Error already logged by ensure_output_dir
            
        run_conversion_stage(abs_input_dir, abs_conversion_output_dir)
        processed_conversion = True

    if not args.skip_segmentation:
        segmentation_input_actual = args.segmentation_input_dir
        if not segmentation_input_actual:
            if args.conversion_output_dir:
                segmentation_input_actual = args.conversion_output_dir
                logging.info(f"Using --conversion-output-dir ({segmentation_input_actual}) as input for segmentation.")
            else:
                parser.error("Missing --segmentation-input-dir and --conversion-output-dir is not available as fallback.")
                return
        
        if not args.segmentation_output_dir:
            parser.error("--segmentation-output-dir is required if --skip-segmentation is not set.")
            return

        abs_segmentation_input_dir = os.path.abspath(segmentation_input_actual)
        abs_segmentation_output_dir = os.path.abspath(args.segmentation_output_dir)

        if not os.path.isdir(abs_segmentation_input_dir):
            logging.error(f"Segmentation input directory '{abs_segmentation_input_dir}' not found.")
            if not processed_conversion and not args.skip_conversion :
                 logging.warning("Perhaps the conversion stage was skipped or failed, and no alternative segmentation input was provided.")
            return
            
        try:
            ensure_output_dir(abs_segmentation_output_dir)
        except Exception: return

        run_segmentation_stage(
            abs_segmentation_input_dir, abs_segmentation_output_dir,
            args.min_silence_len, args.silence_thresh, args.keep_silence,
            args.min_segment_duration, args.max_segment_duration, args.target_split_duration
        )

    logging.info("Audio preprocessing and segmentation script finished.")

if __name__ == "__main__":
    main_cli()
