import argparse
import os
import shutil
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.silence import split_on_silence

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
        print(f"Ensured output directory exists: {directory_path}")
    except OSError as e:
        print(f"Error: Could not create output directory {directory_path}. {e}")
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
        print(f"Info: Converted file {output_filepath} already exists. Skipping conversion.")
        return output_filepath # Return existing path

    try:
        print(f"Converting: {filepath}")
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_channels(1)
        print(f"  Converted to mono.")
        audio = audio.set_frame_rate(TARGET_SAMPLING_RATE)
        print(f"  Resampled to {TARGET_SAMPLING_RATE}Hz.")
        audio.export(output_filepath, format="wav")
        print(f"  Successfully converted and saved to {output_filepath}")
        return output_filepath
    except CouldntDecodeError:
        print(f"Error: Could not decode {filepath}. File might be corrupted or an unsupported format.")
    except FileNotFoundError:
        print(f"Error: ffmpeg/avconv not found for pydub. Please ensure it's installed and in your PATH.")
    except Exception as e:
        print(f"Error converting file {filepath}: {e}")
    return None

def run_conversion_stage(input_dir, conversion_output_dir):
    """
    Recursively scans input_dir for media files and processes them using convert_to_standard_wav.
    """
    print(f"\n--- Starting Conversion Stage ---")
    print(f"Scanning for media files in: {input_dir}")
    print(f"Converted WAVs will be saved to: {conversion_output_dir}")
    
    ensure_output_dir(conversion_output_dir)
    converted_files_count = 0
    error_files_count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            filepath = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            
            if ext.lower() in SUPPORTED_CONVERSION_EXTENSIONS:
                if convert_to_standard_wav(filepath, conversion_output_dir):
                    converted_files_count += 1
                else:
                    error_files_count += 1
            # else:
            #     print(f"Skipping unsupported file for conversion: {filepath}")
    
    print(f"\nConversion Stage Complete.")
    print(f"Successfully converted files: {converted_files_count}")
    print(f"Files failed to convert: {error_files_count}")
    return converted_files_count > 0 or error_files_count > 0 # Return true if any processing happened


def force_split_segment(audio_segment, target_duration_ms, base_output_path, original_filename):
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
        if end_time > duration:
            end_time = duration # Take the remainder
        
        # Ensure the last chunk is not too small (e.g. < 0.5 * target_duration_ms)
        # If it is, it might be better to extend the previous chunk or discard this tiny one.
        # For simplicity now, we keep it if it's > min_duration_ms (e.g. 1s)
        if (end_time - start_time) < 500 and segment_counter > 0 : # Avoid tiny last segment if already split
             print(f"    Force split: Last chunk too small ({end_time - start_time}ms), discarding.")
             break

        chunk = audio_segment[start_time:end_time]
        segment_filename = f"{original_filename}_forced_split_{segment_counter:03d}.wav"
        chunk_output_path = os.path.join(base_output_path, segment_filename)
        
        try:
            chunk.export(chunk_output_path, format="wav")
            print(f"    Force split: Saved chunk {segment_filename} ({len(chunk)}ms)")
            num_chunks_created += 1
            segment_counter += 1
        except Exception as e:
            print(f"    Force split: Error exporting chunk {segment_filename}: {e}")
            
        start_time = end_time
        if start_time >= duration:
            break
            
    return num_chunks_created


def segment_wav_file(wav_filepath, segmentation_output_dir, min_silence_len, silence_thresh, 
                     keep_silence, min_duration_ms, max_duration_ms, target_split_duration_ms):
    """
    Segments a single WAV file based on silence and duration constraints.
    """
    original_filename_no_ext = os.path.splitext(os.path.basename(wav_filepath))[0]
    print(f"\nSegmenting: {wav_filepath}")

    try:
        audio = AudioSegment.from_wav(wav_filepath)
    except Exception as e:
        print(f"  Error loading WAV file {wav_filepath}: {e}")
        return 0

    segments = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
        seek_step=1 # ms, for precision
    )

    if not segments:
        print(f"  No segments found for {wav_filepath} based on silence detection.")
        # If no silence-based segments, consider if the whole file should be force-split if too long
        if len(audio) > max_duration_ms:
             print(f"  Original audio duration ({len(audio)}ms) > max_duration_ms ({max_duration_ms}ms). Attempting force split.")
             return force_split_segment(audio, target_split_duration_ms, segmentation_output_dir, original_filename_no_ext)
        elif len(audio) >= min_duration_ms:
             segment_filename = f"{original_filename_no_ext}_full.wav"
             output_path = os.path.join(segmentation_output_dir, segment_filename)
             audio.export(output_path, format="wav")
             print(f"  Saved full audio as one segment: {segment_filename} ({len(audio)}ms)")
             return 1
        else:
            print(f"  Full audio duration ({len(audio)}ms) < min_duration_ms ({min_duration_ms}ms). Discarding.")
            return 0


    segment_count = 0
    total_segments_saved = 0

    for i, segment in enumerate(segments):
        seg_len = len(segment)
        print(f"  Segment {i}: duration {seg_len}ms")

        if seg_len < min_duration_ms:
            print(f"    Discarded (too short): {seg_len}ms < {min_duration_ms}ms")
            continue
        
        if seg_len > max_duration_ms:
            print(f"    Segment too long ({seg_len}ms > {max_duration_ms}ms). Attempting force split...")
            num_forced_chunks = force_split_segment(segment, target_split_duration_ms, 
                                                    segmentation_output_dir, 
                                                    f"{original_filename_no_ext}_segment_{i:03d}")
            total_segments_saved += num_forced_chunks
        else:
            # Segment is within desired duration range
            segment_filename = f"{original_filename_no_ext}_segment_{segment_count:03d}.wav"
            output_path = os.path.join(segmentation_output_dir, segment_filename)
            try:
                segment.export(output_path, format="wav")
                print(f"    Saved: {segment_filename} ({seg_len}ms)")
                total_segments_saved += 1
            except Exception as e:
                print(f"    Error exporting segment {segment_filename}: {e}")
        segment_count += 1
        
    print(f"  Finished segmenting {wav_filepath}. Total segments saved: {total_segments_saved}")
    return total_segments_saved

def run_segmentation_stage(segmentation_input_dir, segmentation_output_dir, 
                           min_silence_len, silence_thresh, keep_silence, 
                           min_duration_ms, max_duration_ms, target_split_duration_ms):
    """
    Scans segmentation_input_dir for WAV files and segments them.
    """
    print(f"\n--- Starting Segmentation Stage ---")
    print(f"Scanning for WAV files in: {segmentation_input_dir}")
    print(f"Segmented audio clips will be saved to: {segmentation_output_dir}")

    ensure_output_dir(segmentation_output_dir)
    total_segments_generated = 0
    files_processed_count = 0
    files_with_no_segments = 0

    for item in os.listdir(segmentation_input_dir):
        if item.lower().endswith(".wav"):
            filepath = os.path.join(segmentation_input_dir, item)
            if os.path.isfile(filepath):
                files_processed_count +=1
                segments_from_file = segment_wav_file(
                    filepath, segmentation_output_dir,
                    min_silence_len, silence_thresh, keep_silence,
                    min_duration_ms, max_duration_ms, target_split_duration_ms
                )
                if segments_from_file > 0:
                    total_segments_generated += segments_from_file
                else:
                    files_with_no_segments +=1
    
    print(f"\nSegmentation Stage Complete.")
    print(f"Processed {files_processed_count} WAV files.")
    print(f"Total segments generated: {total_segments_generated}")
    print(f"Files resulting in no segments: {files_with_no_segments}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and segment audio/video files for ASR.")
    
    # Conversion stage arguments
    parser.add_argument('--input-dir', type=str, help="Path to the directory containing raw audio/video files for conversion.")
    parser.add_argument('--conversion-output-dir', type=str, help="Path to save converted 16kHz mono WAV files.")
    
    # Segmentation stage arguments
    parser.add_argument('--segmentation-input-dir', type=str, help="Path to directory with 16kHz mono WAVs for segmentation (can be same as --conversion-output-dir).")
    parser.add_argument('--segmentation-output-dir', type=str, help="Path to save segmented audio clips.")
    
    # Control flags
    parser.add_argument('--skip-conversion', action='store_true', help="Skip the initial conversion stage.")
    parser.add_argument('--skip-segmentation', action='store_true', help="Skip the audio segmentation stage.")

    # Segmentation parameters
    parser.add_argument('--min-silence-len', type=int, default=700, help="Min silence length (ms) for split_on_silence. Default: 700ms.")
    parser.add_argument('--silence-thresh', type=int, default=-45, help="Silence threshold (dBFS) for split_on_silence. Default: -45dBFS.")
    parser.add_argument('--keep-silence', type=int, default=250, help="Keep silence (ms) at ends of segments. Default: 250ms.")
    parser.add_argument('--min-segment-duration', type=int, default=2000, help="Minimum duration (ms) for a segment to be kept. Default: 2000ms (2s).")
    parser.add_argument('--max-segment-duration', type=int, default=20000, help="Maximum duration (ms) for a segment before attempting force split. Default: 20000ms (20s).")
    parser.add_argument('--target-split-duration', type=int, default=10000, help="Target duration (ms) for force-splitting long segments. Default: 10000ms (10s).")

    args = parser.parse_args()

    # --- Conversion Stage ---
    if not args.skip_conversion:
        if not args.input_dir or not args.conversion_output_dir:
            parser.error("--input-dir and --conversion-output-dir are required if --skip-conversion is not set.")
        
        abs_input_dir = os.path.abspath(args.input_dir)
        abs_conversion_output_dir = os.path.abspath(args.conversion_output_dir)

        if not os.path.isdir(abs_input_dir):
            print(f"Error: Input directory '{abs_input_dir}' not found or is not a directory.")
            return
        
        try:
            ensure_output_dir(abs_conversion_output_dir)
        except Exception:
            print(f"Exiting due to conversion output directory creation failure.")
            return
            
        print(f"Conversion Input directory: {abs_input_dir}")
        print(f"Conversion Output (16kHz mono WAV) directory: {abs_conversion_output_dir}")
        run_conversion_stage(abs_input_dir, abs_conversion_output_dir)
    else:
        print("Skipping conversion stage as per --skip-conversion flag.")

    # --- Segmentation Stage ---
    if not args.skip_segmentation:
        segmentation_input_actual = args.segmentation_input_dir
        if not segmentation_input_actual and args.conversion_output_dir:
            segmentation_input_actual = args.conversion_output_dir
            print(f"Info: --segmentation-input-dir not specified, using --conversion-output-dir: {segmentation_input_actual}")
        
        if not segmentation_input_actual or not args.segmentation_output_dir:
            parser.error("--segmentation-output-dir is required, and --segmentation-input-dir (or --conversion-output-dir as fallback) must be available if --skip-segmentation is not set.")

        abs_segmentation_input_dir = os.path.abspath(segmentation_input_actual)
        abs_segmentation_output_dir = os.path.abspath(args.segmentation_output_dir)

        if not os.path.isdir(abs_segmentation_input_dir):
            print(f"Error: Segmentation input directory '{abs_segmentation_input_dir}' not found or is not a directory.")
            return
            
        try:
            ensure_output_dir(abs_segmentation_output_dir)
        except Exception:
            print(f"Exiting due to segmentation output directory creation failure.")
            return

        print(f"Segmentation Input directory (16kHz mono WAVs): {abs_segmentation_input_dir}")
        print(f"Segmentation Output directory (segmented clips): {abs_segmentation_output_dir}")
        
        run_segmentation_stage(
            abs_segmentation_input_dir, abs_segmentation_output_dir,
            args.min_silence_len, args.silence_thresh, args.keep_silence,
            args.min_segment_duration, args.max_segment_duration, args.target_split_duration
        )
    else:
        print("Skipping segmentation stage as per --skip-segmentation flag.")

    print("\nAll processing finished.")

if __name__ == "__main__":
    main()
