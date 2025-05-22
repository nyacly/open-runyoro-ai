# Data for Text Generation Model

This directory contains data used for training and evaluating the text generation model.

## Processed Text

-   `processed/train.txt`: This file contains the consolidated text data processed from various sources, ready for model training.

## PDF Document Processing

The system now includes a feature to process PDF documents and extract their text content for use in training the model. This is handled by the `scripts/preprocess_text.py` script.

### Usage

The script `scripts/preprocess_text.py` is used to extract text from all PDF files within a specified directory. The extracted and cleaned text is then appended to an output file.

**Command-line arguments:**

-   `--pdf_dir DIRECTORY`: (Required) Specifies the directory containing the PDF files to be processed.
-   `--output_file FILE_PATH`: (Optional) Specifies the path to the file where the extracted text should be saved. Defaults to `data/processed/train.txt`.

**Example command:**

To process PDF files located in a directory named `my_pdfs` (relative to the project root) and save the output to the default location:

```bash
python scripts/preprocess_text.py --pdf_dir my_pdfs
```

To process PDF files from `path/to/your/pdfs` and save them to a custom file `custom_output.txt`:

```bash
python scripts/preprocess_text.py --pdf_dir path/to/your/pdfs --output_file custom_output.txt
```

### Output

The processed text from the PDF documents will be appended to `data/processed/train.txt` by default, or to the file specified by the `--output_file` argument. Each processed PDF's content will be added as a new entry. If the output file already exists and contains text, new entries will be separated by a newline character.

### Dependency

PDF extraction is performed using the `PyPDF2` library. Ensure that this library is listed in the `requirements.txt` file and installed in your environment.
