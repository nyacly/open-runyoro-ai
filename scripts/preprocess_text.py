import os
import argparse
import logging
import PyPDF2
import re

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content, or None if an error occurs.
    """
    logging.info(f"Extracting text from {pdf_path}")
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                try:
                    reader.decrypt('') # Try with empty password
                    logging.info(f"Decrypted {pdf_path} with empty password.")
                except Exception as e:
                    logging.warning(f"Could not decrypt {pdf_path}. It might be password protected: {e}")
                    return None # Skip password protected files

            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted
                    text += page_text
        logging.info(f"Successfully extracted text from {pdf_path}")
        return text
    except PyPDF2.errors.PdfReadError as e: # More specific error for PyPDF2
        logging.error(f"PyPDF2 error reading PDF {pdf_path}: {e}. The file might be corrupted or not a valid PDF.")
        return None
    except Exception as e:
        logging.error(f"Generic error reading PDF {pdf_path}: {e}")
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

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Could not create output directory {output_dir}: {e}")
            return # Exit if we can't create the output directory

    # Check if output file exists and has content to determine if a leading newline is needed
    prepend_newline_for_next_entry = False
    if os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
        prepend_newline_for_next_entry = True
        logging.info(f"Output file {args.output_file} exists and is not empty. New entries will be prepended with a newline.")


    with open(args.output_file, 'a', encoding='utf-8') as outfile:
        if not os.path.isdir(args.pdf_dir):
            logging.error(f"Provided --pdf_dir '{args.pdf_dir}' is not a directory or does not exist.")
            return

        pdf_files_found = False
        for filename in sorted(os.listdir(args.pdf_dir)): # Sort for deterministic order
            if filename.lower().endswith(".pdf"):
                pdf_files_found = True
                pdf_path = os.path.join(args.pdf_dir, filename)
                logging.info(f"Processing file: {pdf_path}")
                raw_text = extract_text_from_pdf(pdf_path)
                if raw_text:
                    cleaned_text = clean_text(raw_text)
                    if cleaned_text: # Ensure cleaned text is not empty
                        if prepend_newline_for_next_entry:
                            outfile.write('\n') # Add newline before new entry
                        else:
                            # This is the first entry to be written in this run (either new file or empty file)
                            prepend_newline_for_next_entry = True # For subsequent entries
                        
                        outfile.write(cleaned_text)
                        logging.info(f"Appended cleaned text from {filename} to {args.output_file}")
                    else:
                        logging.warning(f"Text from {filename} became empty after cleaning.")
                else:
                    logging.warning(f"No text extracted from {filename} or an error occurred.")
            else:
                logging.info(f"Skipping non-PDF file: {filename}")
        
        if not pdf_files_found:
            logging.warning(f"No PDF files found in directory: {args.pdf_dir}")
    
    logging.info(f"PDF processing completed for directory: {args.pdf_dir}")

if __name__ == "__main__":
    main()
