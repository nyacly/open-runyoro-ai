# Open Runyoro AI 🇷🇼🇺🇬🇹🇿

**Our Vision:** To build open-source AI tools that can read, understand, and speak Runyoro, helping to preserve and promote the language in the digital age.

This project aims to create datasets and models for Natural Language Processing (NLP) and Speech tasks for the Runyoro language.

**Current Goals:**

1.  **Data Collection:**
    *   **Text Corpus:** Collect a diverse and large corpus of written Runyoro.
    *   **Speech Corpus:** Collect transcribed Runyoro audio from native speakers.
2.  **Model Development (Future):**
    *   Text-to-Speech (TTS) for Runyoro.
    *   Automatic Speech Recognition (ASR) for Runyoro.
    *   Machine Translation (e.g., Runyoro <-> English).
    *   Other NLP tools (e.g., part-of-speech taggers, named entity recognizers).

## 🚀 How to Contribute

We welcome contributions from everyone, especially native Runyoro speakers, linguists, and AI/ML developers!

**1. Contributing Data (Most Needed!):**

This is the most crucial part of the project right now. High-quality data is the foundation of good AI models.

*   **Text Data:**
    *   We need plain text files (.txt) containing Runyoro.
    *   Sources can include: books, articles, websites, blogs, proverbs, folk tales, personal writings, etc.
    *   Please ensure the text is in Runyoro and as clean as possible.
    *   **How to submit:** Place your `.txt` files in the `data/text/` directory via a Pull Request. See our [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
*   **Audio Data:**
    *   We need audio recordings (.wav, .mp3, .flac) of spoken Runyoro **along with their accurate transcriptions.**
    *   Ideal audio is clear, with minimal background noise, spoken by a single speaker per file.
    *   **How to submit:**
        1.  Place your audio files in `data/audio/wavs/` (this directory should now exist).
        2.  Create/update a `data/audio/metadata.csv` file with the filename and its transcription. Format: `filename|transcription`. Example: `wavs/runyoro_sentence1.wav|Ekicweka ky'orubazo rwa Runyoro.` (Note: The path in metadata.csv is relative to the `data/audio/` directory).
        3.  Submit via a Pull Request. See our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions, especially regarding audio quality and transcription format.
    *   **Important for Audio:** We use Git LFS for large audio files. Ensure you have it installed (`git lfs install` system-wide or per-user, then the `.gitattributes` file handles repo-specific tracking).

**2. Code Contributions:**
    *   Scripts for data cleaning, preprocessing, model training, etc.
    *   Please open an issue first to discuss your proposed changes.

**3. Linguistic Expertise:**
    *   Help with orthography, grammar, dialect variations, and data validation.

**4. Documentation & Community:**
    *   Improve this README, write tutorials, help answer questions.

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📂 Repository Structure

*   `data/`: Datasets
    *   `data/text/`: Plain text Runyoro corpus.
    *   `data/audio/`: Speech data.
        *   `data/audio/wavs/`: Audio files (e.g., .wav)
        *   `data/audio/metadata.csv`: Transcriptions for audio files (contributors will create/update this with their audio).
*   `scripts/`: Scripts for data processing, training, etc. (Future)
*   `models/`: Trained model files. (Future)
*   `docs/`: Documentation. (Future)

## 💬 Interacting with the Language Model

This project includes a way to interactively chat with the trained Runyoro language model. This allows you to test its capabilities and see what it has learned.

### Prerequisites

1.  **Trained Model:** Ensure you have a trained model. The training script `scripts/train_text.py` saves its output to the `models/text/` directory by default. This chat script expects the model and tokenizer to be present there.
2.  **Dependencies:** Install the necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Chat Script

To start chatting with the model, run the following command from the root directory of the project:

```bash
python scripts/chat.py
```

You can then type your Runyoro phrases, and the model will respond. Type "quit" or "exit" to end the chat session.

## ## Training on Google Colab

This section guides you on how to use the provided Google Colab notebooks for training models.

### 1. Prerequisites
*   A Google account.
*   Access to Google Colab and Google Drive.
*   GPU runtime selected in Colab (Runtime > Change runtime type > GPU).

### 2. Setup Steps

*   **Clone the Repository**:
    *   You can clone the repository into your Google Drive for better persistence of your notebooks and data, or directly into the Colab environment for a temporary session.
    *   Example commands to run in a Colab cell:
        ```bash
        # Option 1: Clone into Google Drive (Recommended)
        # First, mount your Google Drive:
        from google.colab import drive
        drive.mount('/content/drive')

        # Then, navigate to your desired Drive directory (e.g., MyDrive/Colab_Notebooks/your_project) and clone:
        # %cd /content/drive/MyDrive/your_projects_folder/ 
        # !git clone https://github.com/your_username/your_repository.git # Replace with your repo URL
        # %cd your_repository
        
        # Option 2: Clone directly into Colab environment (temporary)
        # !git clone https://github.com/your_username/your_repository.git # Replace with your repo URL
        # %cd your_repository
        ```

*   **Install Dependencies**:
    *   Open either `notebooks/text_training_colab.ipynb` or `notebooks/speechbrain_ssl_training_colab.ipynb`.
    *   Run the first code cell that installs requirements:
        ```python
        !pip install -r requirements.txt
        ```

*   **Prepare Data on Google Drive**:
    *   **For Text Training (`notebooks/text_training_colab.ipynb`)**:
        *   Upload your `train.txt` file to a folder on your Google Drive. For example: `MyDrive/your_project_repo_on_drive/data/processed/train.txt`.
        *   In the notebook, find the "Configuration" cell and update the `BASE_DRIVE_PATH` variable to your project's root path on Drive (e.g., `/content/drive/MyDrive/your_project_repo_on_drive/`) and `DATA_DIR_RELATIVE` to the relative path from the base to your data (e.g., `data/processed`).
    *   **For SpeechBrain SSL Training (`notebooks/speechbrain_ssl_training_colab.ipynb`)**:
        *   Upload your audio data, manifest files (e.g., `train_sb_manifest.json`), and k-means target label files (`<utt_id>_kmeans_labels.npy`) to appropriate folders on your Google Drive.
        *   In the notebook, carefully update the configuration variables in the "Configuration" cell:
            *   `BASE_DRIVE_PATH`: Path to your project's root on Drive (e.g., `/content/drive/MyDrive/your_project_repo_on_drive/`).
            *   `HPARAMS_FILE_REL_PATH`: Relative path *within your cloned repository* to your `hparams_ssl.yaml` (e.g., `runyoro_speech_ai/speechbrain_ssl_training/hparams_ssl.yaml`).
            *   `EXPERIMENT_NAME`: A name for your experiment run (e.g., `ssl_run_01`). Outputs will be saved under a folder with this name.
            *   `DATA_FOLDER_DRIVE`: Full path on Drive where your main audio data and manifest are located (e.g., `/content/drive/MyDrive/your_project_repo_on_drive/data_for_colab/`).
            *   `TRAIN_MANIFEST_REL_PATH`: Relative path *within `DATA_FOLDER_DRIVE`* to your training manifest (e.g., `train_sb_manifest.json` or `manifests/train.json`).
            *   `KMEANS_TARGET_DIR_REL_PATH`: Relative path *within your experiment's output folder on Drive* where k-means targets are expected or will be generated (e.g., `kmeans_frame_labels`).
        *   Refer to the detailed comments and example directory structure provided in the notebook's "Configuration" section for clarity.

### 3. Running the Notebooks

*   Open `notebooks/text_training_colab.ipynb` or `notebooks/speechbrain_ssl_training_colab.ipynb` in Google Colab.
*   **Mount Drive**: Ensure you run the cell that mounts Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
*   **Configure Paths**: Double-check and update all paths in the "Configuration" cell of the chosen notebook. This is crucial for the notebook to locate your data and save outputs correctly to your Google Drive.
*   **Execute Cells**: Run the cells sequentially from top to bottom. The training process will begin, and all outputs (models, checkpoints, logs) will be saved to the directory you specified on Google Drive.

### 4. Important Notes

*   **Output Location**: All model checkpoints, logs, and other outputs will be saved to your Google Drive in the directory specified in the notebook's configuration (`OUTPUT_DIR` for text training, `OUTPUT_FOLDER_DRIVE` for SpeechBrain SSL training).
*   **Resuming Training**: Both Colab notebooks are set up to automatically attempt to resume training from the latest checkpoint if one is found in the designated output directory on your Google Drive.
*   **Resource Limits**: Be mindful of Colab's resource limitations (GPU time, RAM, disk space on the Colab VM). For very long training runs or extremely large models/datasets, you might consider Colab Pro/Pro+ or explore strategies for more distributed data handling and robust checkpointing.
*   **`hparams_ssl.yaml` for SpeechBrain**: The `speechbrain_ssl_training_colab.ipynb` will automatically copy your project's `hparams_ssl.yaml` (specified by `HPARAMS_FILE_REL_PATH`) to the experiment's output directory on Drive (e.g., `BASE_DRIVE_PATH/colab_experiments/EXPERIMENT_NAME/hparams_ssl_colab_exp.yaml`). It then modifies key paths within this *copied* YAML file (`data_folder`, `train_sb_manifest_file`, `output_folder`, `target_label_dir`) to point to your Google Drive locations. For subsequent runs or adjustments, you can directly modify this `hparams_ssl_colab_exp.yaml` file on your Google Drive.

## 📜 License

*   **Code:** Licensed under the [MIT License](LICENSE.md) (You'll need to create this file, or ask Jules to create it with a standard MIT template).
*   **Data:** We encourage contributors to submit data under permissive licenses like Creative Commons (e.g., CC-BY-SA 4.0). Please specify the license for any data you contribute if it's not your original work or if you wish to use a specific license. By default, contributions of original data by contributors are assumed to be under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) unless otherwise specified.

## 💬 Get in Touch

*   **GitHub Issues:** For discussions, bug reports, and feature requests.
*   **(Reachout to openrunyoroai@gmail.com )**

---
*Let's build something amazing for the Runyoro language!*
