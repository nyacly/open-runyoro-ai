import os
import logging
import torch # Added torch import
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Configure basic logging
logging.basicConfig(level=logging.INFO)

def load_model(model_directory="models/text"):
    """
    Loads the GPT-2 model and tokenizer from the specified directory.

    Args:
        model_directory (str): The directory where the model and tokenizer are stored.
                               Defaults to "models/text".

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
               Returns (None, None) if an OSError occurs (e.g., files not found).
    """
    print(f"Loading model and tokenizer from: {model_directory}...") # Added print statement
    logging.info(f"Attempting to load model and tokenizer from: {model_directory}")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
        model = GPT2LMHeadModel.from_pretrained(model_directory)
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except OSError:
        logging.error(f"Error: Model or tokenizer not found in directory: {model_directory}")
        return None, None

def generate_response(model, tokenizer, prompt_text, max_length=100):
    """
    Generates a response from the model based on the prompt_text.

    Args:
        model: The loaded GPT-2 model.
        tokenizer: The loaded GPT-2 tokenizer.
        prompt_text (str): The user's input prompt.
        max_length (int): The maximum length of the generated response.

    Returns:
        str: The decoded response from the model.
    """
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    # Ensure attention_mask is created on the same device as input_ids
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(input_ids.device)
    
    # Generate output sequences
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length + len(input_ids[0]), # Adjust max_length to account for prompt
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        no_repeat_ngram_size=2, # Optional: to prevent repetitive phrases
        early_stopping=True # Optional: to stop generation when EOS token is produced
    )
    
    # Decode the generated tokens, excluding the prompt
    response_text = tokenizer.decode(output_sequences[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response_text

def main():
    """
    Main function to run the chatbot.
    """
    MODEL_DIR = "models/text"
    model, tokenizer = load_model(MODEL_DIR)

    if not model or not tokenizer:
        # Changed logging.error to print for direct user feedback
        print(f"Error: Failed to load model or tokenizer from {MODEL_DIR}. Please ensure the model is trained and files are in the correct location. Exiting.")
        return

    logging.info("Chatbot initialized. Type 'quit' or 'exit' to end the chat.")
    print("Chatbot initialized. Type 'quit' or 'exit' to end the chat.") # Also print to console

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chatbot. Goodbye!") # Added print statement
            logging.info("Exiting chat.")
            break
        
        response = generate_response(model, tokenizer, user_input)
        print(f"Bot: {response}")

if __name__ == '__main__':
    main()
