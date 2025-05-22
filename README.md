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

## 📜 License

*   **Code:** Licensed under the [MIT License](LICENSE.md) (You'll need to create this file, or ask Jules to create it with a standard MIT template).
*   **Data:** We encourage contributors to submit data under permissive licenses like Creative Commons (e.g., CC-BY-SA 4.0). Please specify the license for any data you contribute if it's not your original work or if you wish to use a specific license. By default, contributions of original data by contributors are assumed to be under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) unless otherwise specified.

## 💬 Get in Touch

*   **GitHub Issues:** For discussions, bug reports, and feature requests.
*   **(Reachout to openrunyoroai@gmail.com )**

---
*Let's build something amazing for the Runyoro language!*
