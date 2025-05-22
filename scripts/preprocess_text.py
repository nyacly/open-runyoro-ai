import os
import argparse
import logging
import PyPDF2
import re
from pdf2image import convert_from_path
from pytesseract import image_to_string
from PyPDF2 import PdfReader
from tqdm import tqdm
import pytesseract
from multiprocessing import Pool

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from pdf2image import convert_from_path

def ocr_page(image):
    return image_to_string(image)

def extract_text_from_pdf(pdf_path, max_pages=50):
    """
    Extract text from a PDF, including text from images using OCR with multiprocessing.
    """
    try:
        text = ""

        # Step 1: Extract text from text-based PDFs using PyPDF2
        reader = PdfReader(pdf_path)
        for page in tqdm(reader.pages[:max_pages], desc=f"Processing text-based pages in {os.path.basename(pdf_path)}"):
            if page.extract_text():
                text += page.extract_text() + "\n"

        # Step 2: Use OCR for image-based PDFs with multiprocessing
        images = convert_from_path(pdf_path)
        with Pool(processes=4) as pool:  # Adjust the number of processes as needed
            ocr_results = pool.map(ocr_page, images[:max_pages])
        text += "\n".join(ocr_results)

        return text
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
        return None

def clean_text(text):
    """
    Performs basic text cleaning.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    # Attempt to fix words broken by hyphenation at line breaks
    # This regex looks for a word character, a hyphen, optional whitespace, a newline, optional whitespace, and another word character
    text = re.sub(r'(\w)-(\s*)\n(\s*)(\w)', r'\1\4', text) 
    # Simpler hyphen removal if the above doesn't catch all cases (e.g. hyphen at end of line without immediate word continuation)
    text = re.sub(r'-\n', '', text)
    # Remove excessive newlines (replace multiple newlines with a single one)
    text = re.sub(r'\n\s*\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def main():
    parser = argparse.ArgumentParser(description="Extracts text from PDF files in a directory and saves it to a file.")
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF files.")
    parser.add_argument("--output_file", default="data/processed/train.txt", help="Output file to save extracted text (default: data/processed/train.txt).")
    args = parser.parse_args()

    logging.info(f"Starting PDF processing. Input directory: {args.pdf_dir}, Output file: {args.output_file}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Could not create output directory {output_dir}: {e}")
            return  # Exit if we can't create the output directory

    # Check if output file exists and has content to determine if a leading newline is needed
    prepend_newline_for_next_entry = False
    if os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
        prepend_newline_for_next_entry = True
        logging.info(f"Output file {args.output_file} exists and is not empty. New entries will be prepended with a newline.")

    # Ensure the input directory exists
    if not os.path.isdir(args.pdf_dir):
        logging.error(f"Provided --pdf_dir '{args.pdf_dir}' is not a directory or does not exist.")
        return

    # Process all PDF files in the directory
    pdf_files = [f for f in sorted(os.listdir(args.pdf_dir)) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logging.warning(f"No PDF files found in directory: {args.pdf_dir}")
        return

    with open(args.output_file, 'a', encoding='utf-8') as outfile:
        for filename in pdf_files:
            pdf_path = os.path.join(args.pdf_dir, filename)
            logging.info(f"Processing file: {pdf_path}")
            raw_text = extract_text_from_pdf(pdf_path)
            if raw_text:
                cleaned_text = clean_text(raw_text)
                if cleaned_text:  # Ensure cleaned text is not empty
                    if prepend_newline_for_next_entry:
                        outfile.write('\n')  # Add newline before new entry
                    outfile.write(cleaned_text)
                    prepend_newline_for_next_entry = True  # For subsequent entries
                    logging.info(f"Appended cleaned text from {filename} to {args.output_file}")
                else:
                    logging.warning(f"Text from {filename} became empty after cleaning.")
            else:
                logging.warning(f"No text extracted from {filename} or an error occurred.")

    logging.info(f"PDF processing completed for directory: {args.pdf_dir}")

if __name__ == "__main__":
    main()
