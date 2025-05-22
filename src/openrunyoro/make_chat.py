import argparse
import logging
import json
import re
import spacy
import os

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist (should be there from setup)
if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True) # exist_ok=True for safety

# File handler for logging
file_handler = logging.FileHandler("logs/make_chat.log", mode='w', encoding='utf-8')
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

def main():
    """
    Main function to parse arguments, load models, and prepare for chat generation.
    """
    parser = argparse.ArgumentParser(description="Convert cleaned text into a chat JSONL format.")
    parser.add_argument(
        "--in",
        dest="input_path",  # Using dest to avoid conflict with 'in' keyword
        required=True,
        help="Path to the input cleaned text file (e.g., data/clean/book.txt).",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        required=True,
        help="Path for the output JSONL file (e.g., data/processed/book_chat.jsonl).",
    )
    parser.add_argument(
        "--template",
        default="alpaca",
        help="The template to use for formatting chat turns (default 'alpaca'). Currently not used.",
    )
    parser.add_argument(
        "--lang_model",
        default="xx_sent_wiki_sm", # As discussed, using xx_sent_wiki_sm
        help="SpaCy language model to use for sentence splitting (default 'xx_sent_wiki_sm').",
    )

    args = parser.parse_args()

    logger.info(f"Parsed arguments: Input file: '{args.input_path}', Output file: '{args.output_path}', "
                f"Template: '{args.template}', SpaCy model: '{args.lang_model}'")

    # Load SpaCy model
    nlp = None
    try:
        logger.info(f"Loading SpaCy model: {args.lang_model}...")
        nlp = spacy.load(args.lang_model)
        logger.info(f"SpaCy model '{args.lang_model}' loaded successfully.")
    except OSError as e:
        logger.error(f"Failed to load SpaCy model '{args.lang_model}'. Error: {e}")
        logger.error(f"Please make sure the model is downloaded. You can try: "
                     f"python -m spacy download {args.lang_model}")
        return # Exit if model can't be loaded

    # Read entire file content
    text_content = ""
    try:
        logger.info(f"Reading content from input file: {args.input_path}")
        with open(args.input_path, "r", encoding="utf-8") as f_in:
            text_content = f_in.read()
        logger.info(f"Successfully read {len(text_content)} characters from '{args.input_path}'.")
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_path}")
        return 
    except Exception as e:
        logger.error(f"An error occurred while reading the input file '{args.input_path}': {e}")
        return

    if not text_content.strip():
        logger.warning(f"Input file '{args.input_path}' is empty or contains only whitespace. No sentences to process.")
        # TODO: Decide if an empty output file should be created or not. For now, just exit.
        return

    # Sentence Splitting
    logger.info("Processing text with SpaCy for sentence splitting...")
    doc = nlp(text_content)
    sentences_text = []
    for sent in doc.sents:
        stripped_sent = sent.text.strip()
        if stripped_sent: # Only add non-empty sentences
            sentences_text.append(stripped_sent)
    
    logger.info(f"Found {len(sentences_text)} sentences after SpaCy processing.")

    # Heuristic Detection
    # Dialogue: r"^\s*([A-Za-z\s]+(?:’s)?[:-])\s*(.+)"
    # Vocabulary: r"^\s*•?\s*([a-z’]+)\s*–\s*(.+)"
    # The problem description's regex for dialogue seems to prefer English alphabet for speaker.
    # For Runyoro/Rutooro, a more general speaker prefix might be needed, but using the provided one.
    # Python's re module handles Unicode by default in Python 3.
    dialogue_regex = re.compile(r"^\s*([\w\s]+(?:’s)?[:-])\s*(.+)") # Using \w for speaker to be more general
    vocab_regex = re.compile(r"^\s*•?\s*([a-zA-Z’']+)\s*[-–—]\s*(.+)") # Allow different dashes, ensure word part matches typical word chars

    dialogue_segments = []
    vocabulary_segments = []
    other_segments = []

    for sentence_str in sentences_text:
        dialogue_match = dialogue_regex.match(sentence_str)
        if dialogue_match:
            speaker = dialogue_match.group(1).strip()
            utterance = dialogue_match.group(2).strip()
            if speaker and utterance: # Ensure both parts are non-empty
                dialogue_segments.append({'speaker': speaker, 'utterance': utterance})
            else: # If parts are empty after strip, treat as other
                other_segments.append(sentence_str)
            continue # Move to next sentence

        vocab_match = vocab_regex.match(sentence_str)
        if vocab_match:
            word = vocab_match.group(1).strip()
            definition = vocab_match.group(2).strip()
            if word and definition: # Ensure both parts are non-empty
                 vocabulary_segments.append({'word': word, 'definition': definition})
            else: # If parts are empty after strip, treat as other
                other_segments.append(sentence_str)
            continue # Move to next sentence
        
        # If neither matches, it's an "other" segment
        other_segments.append(sentence_str)

    logger.info(f"Detected {len(dialogue_segments)} dialogue segments.")
    logger.info(f"Detected {len(vocabulary_segments)} vocabulary segments.")
    logger.info(f"Detected {len(other_segments)} other segments (unmatched).")

    all_chat_turns = []

    # 1. Process Dialogue Segments
    current_speaker = None
    current_role = "user" # Start with user
    if dialogue_segments:
        # Initialize with the first speaker
        current_speaker = dialogue_segments[0]['speaker']
        # First turn is always user, then we alternate based on speaker change
        # This means the first speaker is 'user', second is 'assistant', third is 'user' again if different from second.
        
        # Determine initial role for the first speaker based on a consistent starting point.
        # Let's decide the first speaker in the entire dialogue sequence is 'user'.
        # The role will then toggle when the speaker changes.
        
        # Simpler: first speaker encountered is user, next different speaker is assistant, etc.
        speaker_to_role_map = {}
        
        for segment in dialogue_segments:
            speaker = segment['speaker']
            utterance = segment['utterance']

            if speaker not in speaker_to_role_map:
                if len(speaker_to_role_map) == 0: # First unique speaker
                    speaker_to_role_map[speaker] = "user"
                elif len(speaker_to_role_map) == 1: # Second unique speaker
                    # Ensure it's different from the first role
                    existing_role = list(speaker_to_role_map.values())[0]
                    speaker_to_role_map[speaker] = "assistant" if existing_role == "user" else "user"
                else: # More than two speakers, alternate based on previous turn's role for simplicity or assign 'user'
                      # For now, let's default additional speakers to 'user' or cycle through user/assistant
                      # This heuristic is imperfect for >2 speakers.
                      # A simple alternating logic based on previous turn if speaker changes
                    if all_chat_turns and all_chat_turns[-1]['role'] == "user":
                         speaker_to_role_map[speaker] = "assistant"
                    else:
                         speaker_to_role_map[speaker] = "user"

            assigned_role = speaker_to_role_map[speaker]
            all_chat_turns.append({"role": assigned_role, "content": utterance})
        logger.info(f"Generated {len(all_chat_turns)} turns from dialogue segments.")


    # 2. Process Vocabulary Segments
    vocab_turns_start_index = len(all_chat_turns)
    for segment in vocabulary_segments:
        word = segment['word']
        definition = segment['definition']
        all_chat_turns.append({"role": "user", "content": f"What does '{word}' mean?"})
        all_chat_turns.append({"role": "assistant", "content": definition})
        all_chat_turns.append({"role": "user", "content": f"How do you say '{definition}' in Runyoro?"}) # Assuming Runyoro
        all_chat_turns.append({"role": "assistant", "content": word})
    logger.info(f"Generated {len(all_chat_turns) - vocab_turns_start_index} turns from vocabulary segments.")

    # 3. Process Other Segments
    other_turns_start_index = len(all_chat_turns)
    for segment_text in other_segments:
        all_chat_turns.append({"role": "user", "content": f"Explain this: {segment_text}"})
        # Truncate for assistant's response summary
        summary_snippet = segment_text[:75] + "..." if len(segment_text) > 75 else segment_text
        all_chat_turns.append({"role": "assistant", "content": f"This appears to be a section about: {summary_snippet} What would you like to discuss or understand about it?"})
    logger.info(f"Generated {len(all_chat_turns) - other_turns_start_index} turns from other segments.")

    logger.info(f"Total chat turns generated: {len(all_chat_turns)}")

    # Group into Conversations (max 10 turns per conversation)
    final_conversations_list_of_lists = []
    if all_chat_turns:
        for i in range(0, len(all_chat_turns), 10):
            final_conversations_list_of_lists.append(all_chat_turns[i:i + 10])
    
    logger.info(f"Grouped turns into {len(final_conversations_list_of_lists)} conversations.")

    # Output to JSONL
    try:
        logger.info(f"Writing {len(final_conversations_list_of_lists)} conversations to JSONL file: {args.output_path}")
        with open(args.output_path, "w", encoding="utf-8") as f_out:
            for conversation in final_conversations_list_of_lists:
                f_out.write(json.dumps(conversation) + "\n")
        logger.info("Successfully written conversations to JSONL file.")
    except IOError as e:
        logger.error(f"Failed to write to output file '{args.output_path}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSONL writing: {e}")

if __name__ == "__main__":
    main()
