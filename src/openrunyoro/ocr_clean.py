import argparse
import os
import glob
import logging
import io
import re
import unicodedata
import multiprocessing # For future use
from pdfminer.high_level import extract_text_to_fp
from PIL import Image # For image type checks, and potentially for pdf2image
import pytesseract
import pdf2image # To convert PDF to images

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist (should be there from setup)
if not os.path.exists("logs"):
    os.makedirs("logs")

# File handler for logging
file_handler = logging.FileHandler("logs/ocr_clean.log", mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler for logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


def get_avg_confidence(ocr_data):
    """
    Calculate average confidence from pytesseract data.
    Skips words with confidence -1 (often non-text elements or headers).
    Returns a tuple (average_confidence, warning_message_or_none).
    """
    confidences = []
    warning_message = None
    lines = ocr_data.strip().split('\n')
    if len(lines) < 2: # Header + at least one data line
        return 0, warning_message
    
    header = lines[0].split('\t')
    try:
        conf_idx = header.index('conf')
    except ValueError:
        warning_message = "Could not find 'conf' column in tesseract data. Cannot calculate confidence."
        return 0, warning_message

    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) == len(header):
            try:
                confidence = float(parts[conf_idx])
                if confidence != -1: # Valid confidence value
                    confidences.append(confidence)
            except ValueError:
                # Could not parse confidence for this line
                pass
    
    if not confidences:
        return 0, warning_message
    return sum(confidences) / len(confidences), warning_message


def process_single_file(filepath: str, lang: str) -> tuple[str, list[str]]:
    """
    Processes a single file (PDF or image) for text extraction and OCR.
    Returns a tuple containing the extracted text and a list of log messages.
    """
    file_text_buffer = io.StringIO()
    log_messages = []
    
    file_basename = os.path.basename(filepath)
    log_messages.append(f"INFO: Processing file: {filepath}")

    try:
        file_extension = os.path.splitext(filepath)[1].lower()

        if file_extension == ".pdf":
            extracted_text_pdfminer = io.StringIO()
            pdfminer_success = False
            try:
                with open(filepath, "rb") as f_pdf:
                    extract_text_to_fp(f_pdf, extracted_text_pdfminer)
                
                pdfminer_text = extracted_text_pdfminer.getvalue()
                if len(pdfminer_text.strip()) > 100: # Arbitrary threshold
                    log_messages.append(f"INFO: Successfully extracted text from PDF '{filepath}' using pdfminer.")
                    file_text_buffer.write(pdfminer_text)
                    file_text_buffer.write("\n\n")
                    pdfminer_success = True
                else:
                    log_messages.append(f"WARNING: pdfminer extracted very little text ({len(pdfminer_text.strip())} chars) from '{filepath}'. Attempting OCR fallback.")
            except Exception as e_pdfminer:
                log_messages.append(f"WARNING: pdfminer extraction failed for '{filepath}': {e_pdfminer}. Falling back to OCR.")

            if not pdfminer_success:
                try:
                    log_messages.append(f"INFO: Converting PDF '{filepath}' to images for OCR...")
                    images = pdf2image.convert_from_path(filepath, dpi=300)
                    if not images:
                        log_messages.append(f"ERROR: pdf2image returned no images for '{filepath}'. Skipping OCR for this file.")
                    else:
                        for i, image in enumerate(images):
                            page_num = i + 1
                            log_messages.append(f"INFO: Performing OCR on page {page_num} of '{filepath}'...")
                            try:
                                ocr_text_page = pytesseract.image_to_string(image, lang=lang)
                                ocr_data_page = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.STRING)
                                avg_conf, conf_warn = get_avg_confidence(ocr_data_page)
                                if conf_warn:
                                    log_messages.append(f"WARNING: {conf_warn} (Page {page_num}, File {filepath})")
                                log_messages.append(f"INFO: OCR for page {page_num} of '{filepath}' completed. Average confidence: {avg_conf:.2f}%")
                                file_text_buffer.write(f"--- OCR Page {page_num} from {file_basename} (Confidence: {avg_conf:.2f}%) ---\n")
                                file_text_buffer.write(ocr_text_page)
                                file_text_buffer.write("\n\n")
                            except pytesseract.TesseractError as e_ocr_page:
                                log_messages.append(f"ERROR: Tesseract OCR failed for page {page_num} of '{filepath}': {e_ocr_page}")
                            except Exception as e_page_proc:
                                log_messages.append(f"ERROR: Error processing page {page_num} of '{filepath}' with Tesseract: {e_page_proc}")
                except pdf2image.exceptions.PDFInfoNotInstalledError:
                    log_messages.append("ERROR: pdfinfo from poppler-utils is not installed. PDF processing via OCR will fail. Please install poppler-utils.")
                    log_messages.append("ERROR: On Debian/Ubuntu: sudo apt-get install poppler-utils")
                except pdf2image.exceptions.PDFPageCountError:
                     log_messages.append(f"ERROR: Could not get page count for PDF '{filepath}'. Is it a valid PDF? Skipping OCR.")
                except pdf2image.exceptions.PDFSyntaxError:
                    log_messages.append(f"ERROR: PDF syntax error in '{filepath}'. Skipping OCR.")
                except Exception as e_pdf_ocr:
                    log_messages.append(f"ERROR: Failed to OCR PDF '{filepath}': {e_pdf_ocr}")

        elif file_extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            log_messages.append(f"INFO: Performing OCR on image file: {filepath}")
            try:
                ocr_text_img = pytesseract.image_to_string(filepath, lang=lang)
                ocr_data_img = pytesseract.image_to_data(filepath, lang=lang, output_type=pytesseract.Output.STRING)
                avg_conf, conf_warn = get_avg_confidence(ocr_data_img)
                if conf_warn:
                    log_messages.append(f"WARNING: {conf_warn} (File {filepath})")
                log_messages.append(f"INFO: OCR for image '{filepath}' completed. Average confidence: {avg_conf:.2f}%")
                file_text_buffer.write(f"--- OCR from Image {file_basename} (Confidence: {avg_conf:.2f}%) ---\n")
                file_text_buffer.write(ocr_text_img)
                file_text_buffer.write("\n\n")
            except pytesseract.TesseractError as e_img_ocr:
                log_messages.append(f"ERROR: Tesseract OCR failed for image '{filepath}': {e_img_ocr}")
            except FileNotFoundError:
                log_messages.append(f"ERROR: Image file not found: {filepath}")
            except Exception as e_img_proc:
                log_messages.append(f"ERROR: Error processing image '{filepath}' with Tesseract: {e_img_proc}")
        
        else:
            log_messages.append(f"WARNING: Unsupported file type: '{filepath}'. Skipping.")

    except Exception as e_file_proc:
        log_messages.append(f"ERROR: Failed to process file '{filepath}': {e_file_proc}")
    
    return file_text_buffer.getvalue(), log_messages


def clean_text(text: str) -> str:
    """
    Cleans the extracted text by normalizing unicode, removing control characters,
    normalizing whitespace, and dropping short lines.
    """
    if not text:
        return ""

    # 1. Unicode Normalization
    text = unicodedata.normalize('NFKC', text)

    # 2. Strip control characters (common C0 and C1 controls)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # 3. Whitespace normalization
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Trim leading/trailing spaces from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    # Filter out lines with fewer than 3 words
    lines = [line for line in lines if len(line.split()) >= 3]
    # Join lines back, then normalize multiple newlines to a single newline
    text = '\n'.join(lines)
    text = re.sub(r'\n\s*\n+', '\n', text) # Multiple newlines (with optional space in between) to one
    
    # Final trim of the whole text
    text = text.strip()

    return text


def main():
    """
    Main function to parse arguments and orchestrate the OCR cleaning process.
    """
    parser = argparse.ArgumentParser(description="Clean OCR text from PDF/image files.")
    parser.add_argument(
        "--pdf", # Retaining name for consistency, but now handles PDF/PNG/JPG
        nargs="+",
        required=True,
        help="Glob pattern(s) for input PDF/PNG/JPG files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path for the output UTF-8 text file.",
    )
    parser.add_argument(
        "--lang",
        default="run",
        help="Language for Tesseract OCR (default 'run').",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count(), # Default to number of CPU cores
        help=f"Number of threads for parallel processing (default: {os.cpu_count()}).",
    )

    args = parser.parse_args()

    logger.info(f"Parsed arguments: Input patterns: {args.pdf}, Output file: {args.out}, Language: {args.lang}, Threads: {args.threads}")

    all_input_files = []
    for pattern in args.pdf:
        expanded_files = glob.glob(pattern)
        if not expanded_files:
            logger.warning(f"Input pattern '{pattern}' did not match any files.")
        all_input_files.extend(expanded_files)
    
    if not all_input_files:
        logger.error("No input files found after expanding glob patterns. Exiting.")
        return

    logger.info(f"Found {len(all_input_files)} input file(s) to process.")

    # Prepare arguments for starmap
    process_args = [(filepath, args.lang) for filepath in all_input_files]

    accumulated_text_buffer = io.StringIO()
    
    logger.info(f"Starting parallel processing with {args.threads} threads...")
    with multiprocessing.Pool(processes=args.threads) as pool:
        results = pool.starmap(process_single_file, process_args)

    logger.info("Parallel processing finished. Aggregating results...")

    for file_text, log_list in results:
        accumulated_text_buffer.write(file_text)
        for log_entry in log_list:
            level, _, message = log_entry.partition(': ') # Simple split, assumes "LEVEL: Message"
            if level == "INFO":
                logger.info(message)
            elif level == "WARNING":
                logger.warning(message)
            elif level == "ERROR":
                logger.error(message)
            else: # Fallback for messages not fitting LEVEL: structure
                logger.info(log_entry)


    raw_text = accumulated_text_buffer.getvalue()
    logger.info(f"Raw text accumulated. Length: {len(raw_text)} characters.")

    logger.info("Cleaning accumulated text...")
    cleaned_text = clean_text(raw_text)
    logger.info(f"Cleaned text generated. Length: {len(cleaned_text)} characters.")

    # Write cleaned text to output file
    try:
        with open(args.out, "w", encoding="utf-8") as f_out:
            f_out.write(cleaned_text)
        logger.info(f"Cleaned text successfully written to '{args.out}'")
    except IOError as e:
        logger.error(f"Failed to write cleaned text to '{args.out}': {e}")
        # Optionally, print to stdout if file write fails
        # print("\n" + "="*80 + "\nCLEANED TEXT (File Write Failed):\n" + "="*80 + "\n")
        # print(cleaned_text)
        # print("\n" + "="*80 + "\n")


    # Calculate and log word count
    word_count = len(cleaned_text.split())
    logger.info(f"Total word count of cleaned text: {word_count}")
    print(f"Total word count of cleaned text: {word_count}") # Also to stdout as requested

    logger.info("Finished processing all files.")

if __name__ == "__main__":
    main()
