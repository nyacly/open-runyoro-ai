# Jupyter Notebooks for Runyoro Speech AI Project

This directory contains Jupyter notebooks for various tasks in the Runyoro Speech AI project.

## Available Notebooks:

*   **`generate_kmeans_targets_colab.ipynb`**: A Colab notebook for generating K-means targets from audio features. This is a prerequisite for HuBERT-style SSL training.
*   **`speechbrain_ssl_training_colab.ipynb`**: (DEPRECATED) Original Colab notebook for SpeechBrain SSL training. This notebook had issues with JSON formatting and opening in Colab.
*   **`speechbrain_ssl_training_colab_v2.ipynb`**: **(Recommended for Colab SSL Training)** A revised and cleaned-up Colab notebook for performing Self-Supervised Learning (SSL) model training using SpeechBrain, specifically adapted for the HuBERT-style masked prediction objective. This notebook addresses the JSON corruption issues found in the original and provides a structured way to run SSL training in a Google Colab environment, especially with GPU acceleration (e.g., A100). It guides through setup, configuration of paths on Google Drive, and execution of the training pipeline.
*   **`text_training.ipynb`**: Notebook for text model training (details to be added).
*   **`text_training_colab.ipynb`**: Colab version for text model training (details to be added).
*   **`tts_training.ipynb`**: Notebook for Text-to-Speech model training (details to be added).
*   **`youtube_data_ingestion_colab.ipynb`**: A Colab notebook to assist with ingesting data from YouTube, likely involving downloading and initial processing.

Please refer to the individual notebooks for specific instructions and execution details.
