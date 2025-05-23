# Runyoro Speech AI Project

This project aims to develop robust Artificial Intelligence capabilities for the Runyoro language, starting with Automatic Speech Recognition (ASR) and laying the groundwork for future speech generation (Text-to-Speech) and natural language understanding (NLU).

## Project Phases (High-Level)

1.  **Data Ingestion & Preprocessing:** Automated pipeline for acquiring and preparing Runyoro audio data.
2.  **Self-Supervised Learning (SSL) Model Training:** Pre-training a model on unlabeled Runyoro audio.
3.  **ASR Fine-tuning:** Fine-tuning the SSL model for speech recognition.
4.  **Inference System:** CLI/API for transcribing Runyoro audio.
5.  **Documentation & Scalability:** Comprehensive docs and cloud scaling plan.

## Setup

(Instructions to be added once environment setup is stable and `requirements.txt` is finalized.)

## Data Ingestion Pipeline

A detailed guide on the data ingestion and preprocessing pipeline, including setup, usage of the main orchestration script (`data_ingestion/main_ingest.py`), command-line arguments, and troubleshooting, can be found in:

*   **[Data Ingestion and Preprocessing Documentation](./docs/data_ingestion.md)**

This pipeline handles:
*   Downloading audio from YouTube.
*   Ingesting local audio/video files.
*   Standardizing audio to 16kHz mono WAV format.
*   Segmenting audio based on silence and duration.
*   Generating a manifest file for ASR model training.

---

Further sections on SSL training, ASR fine-tuning, etc., will be added as the project progresses.
