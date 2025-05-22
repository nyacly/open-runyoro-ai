import subprocess
import pytest
from pathlib import Path
import os

# Attempt to import reportlab, but have a fallback or skip if not available
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Attempt to import Pillow for fallback image generation
try:
    from PIL import Image, ImageDraw, ImageFont
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# Helper to create a sample PDF
def create_sample_pdf(path: Path, text1: str, text2: str):
    if not REPORTLAB_AVAILABLE:
        pytest.skip("ReportLab is not available, skipping PDF generation.")
    
    c = canvas.Canvas(str(path), pagesize=letter)
    # Page 1
    c.drawString(1 * inch, 10 * inch, text1)
    c.showPage()
    # Page 2
    c.drawString(1 * inch, 10 * inch, text2)
    c.save()

# Helper to create a sample PNG image
def create_sample_png(path: Path, text: str):
    if not PILLOW_AVAILABLE:
        pytest.skip("Pillow is not available, skipping PNG generation.")
    
    img_width = 400
    img_height = 100
    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)
    try:
        # Try to load a common font, fall back to default if not found
        font = ImageFont.truetype("DejaVuSans", 20) # DejaVuSans is common on Linux
    except IOError:
        font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0,0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (img_width - text_width) / 2
    y = (img_height - text_height) / 2
    draw.text((x, y), text, fill="black", font=font)
    img.save(path)


@pytest.fixture(scope="module", autouse=True)
def ensure_logs_dir():
    Path("logs").mkdir(exist_ok=True)

def test_ocr_clean_pdf_processing(tmp_path):
    if not REPORTLAB_AVAILABLE:
        pytest.skip("ReportLab is not available, cannot run PDF processing test.")

    sample_pdf_path = tmp_path / "sample.pdf"
    output_txt_path = tmp_path / "output.txt"
    
    page1_text = "This is the first page for PDF test. It has some unique text."
    page2_text = "This is the second page for PDF test. More specific content here."
    
    create_sample_pdf(sample_pdf_path, page1_text, page2_text)

    cmd = [
        "python", "-m", "src.openrunyoro.ocr_clean", # Adjusted path to run as module
        "--pdf", str(sample_pdf_path),
        "--out", str(output_txt_path),
        "--lang", "eng", # Use 'eng' for reliable testing in CI
        "--threads", "1" # Use 1 thread for deterministic testing output if order matters
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

    assert result.returncode == 0, f"ocr_clean.py failed with stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_txt_path.exists(), "Output text file was not created."
    
    content = output_txt_path.read_text(encoding="utf-8")
    
    # Check for presence of text from both pages (after cleaning)
    # The cleaning process might alter the text, so check for keywords
    assert "first page" in content.lower(), "Text from first page not found in output."
    assert "second page" in content.lower(), "Text from second page not found in output."
    assert "unique text" in content.lower(), "Unique text from first page not found."
    assert "specific content" in content.lower(), "Specific content from second page not found."

    # A general check for a reasonable amount of text, adjusted for cleaning
    # The cleaning function drops lines with < 3 words.
    # "This is the first page for PDF test." -> 7 words
    # "It has some unique text." -> 5 words
    # "This is the second page for PDF test." -> 7 words
    # "More specific content here." -> 4 words
    # Total expected characters will be less than raw, but should be substantial.
    # Let's aim for something like 50 characters after cleaning.
    assert len(content) >= 50, f"Output content is too short ({len(content)} chars). Content:\n{content}"

    log_file = Path("logs/ocr_clean.log")
    assert log_file.exists(), "Log file 'logs/ocr_clean.log' was not created."
    
    log_content = log_file.read_text(encoding="utf-8")
    assert "Total word count of cleaned text" in log_content or "Total word count of cleaned text" in result.stdout, \
        "Word count message not found in logs or stdout."
    assert "Successfully extracted text from PDF" in log_content or "pdfminer extracted very little text" in log_content or "Performing OCR on page" in log_content, \
        "Expected PDF processing log messages not found."

def test_ocr_clean_png_processing(tmp_path):
    if not PILLOW_AVAILABLE:
        pytest.skip("Pillow is not available, cannot run PNG processing test.")

    sample_png_path = tmp_path / "sample.png"
    output_txt_path = tmp_path / "output_png.txt"
    
    image_text = "Sample text for PNG image OCR."
    create_sample_png(sample_png_path, image_text)

    cmd = [
        "python", "-m", "src.openrunyoro.ocr_clean", # Adjusted path
        "--pdf", str(sample_png_path), # Argument name is --pdf, but it handles images
        "--out", str(output_txt_path),
        "--lang", "eng",
        "--threads", "1"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

    assert result.returncode == 0, f"ocr_clean.py failed with stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_txt_path.exists(), "Output text file was not created for PNG test."
    
    content = output_txt_path.read_text(encoding="utf-8")
    
    assert "sample text" in content.lower(), "Text from PNG not found in output."
    assert "png image ocr" in content.lower(), "Text from PNG not found in output."
    assert len(content) >= 15, f"Output content from PNG is too short ({len(content)} chars). Content:\n{content}" # "Sample text for PNG image OCR." is 5 words

    log_file = Path("logs/ocr_clean.log")
    assert log_file.exists(), "Log file 'logs/ocr_clean.log' was not created (PNG test)."
    
    log_content = log_file.read_text(encoding="utf-8")
    assert "Total word count of cleaned text" in log_content or "Total word count of cleaned text" in result.stdout, \
        "Word count message not found in logs or stdout (PNG test)."
    assert f"OCR for image '{str(sample_png_path)}' completed" in log_content, \
        "Expected PNG OCR log messages not found."

# A very basic test to ensure the script runs with --help
def test_ocr_clean_help(tmp_path):
    cmd = ["python", "-m", "src.openrunyoro.ocr_clean", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, f"ocr_clean.py --help failed: {result.stderr}"
    assert "usage: ocr_clean.py" in result.stdout or "usage: __main__.py" in result.stdout
    assert "--pdf" in result.stdout
    assert "--out" in result.stdout
    assert "--lang" in result.stdout
    assert "--threads" in result.stdout
