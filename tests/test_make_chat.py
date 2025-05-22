import subprocess
import pytest
from pathlib import Path
import json
import os

# Content for the sample input file
SAMPLE_TEXT_CONTENT = """
SpeakerA: Hello from Speaker A.
SpeakerB: Greetings from Speaker B!
This is a general statement about something interesting.
omwana – child
•enté – cow
SpeakerA: How are you doing today, B?
SpeakerB: I am doing well, A. And you?
Another piece of information for context.
amasaka – sorghum
igufa – bone
SpeakerA: I'm also good.
SpeakerB: That is great to hear.
Yet another explanatory line.
embwa – dog
omwaka – year
SpeakerA: Let's talk about Runyoro.
SpeakerB: Yes, let's.
This could be a section on verbs.
omutwe – head
ekitabo – book
SpeakerA: I like learning languages.
SpeakerB: Me too! It's fun.
Final sentence for this test.
""" # Approx 23 actual content lines

@pytest.fixture(scope="module", autouse=True)
def ensure_logs_dir_for_make_chat():
    # Separate fixture name to avoid conflict if other tests use a similar one
    Path("logs").mkdir(exist_ok=True)

def test_make_chat_basic(tmp_path):
    # 1. Setup
    sample_input_path = tmp_path / "sample_input.txt"
    output_jsonl_path = tmp_path / "output.jsonl"
    
    with open(sample_input_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_TEXT_CONTENT)

    # The spaCy model 'xx_sent_ud_sm' was installed in a previous step.
    # If this test were run in a completely fresh environment,
    # one might add:
    # subprocess.run(["python", "-m", "spacy", "download", "xx_sent_ud_sm"], check=True, capture_output=True)
    # However, for this agent, it's assumed to be present from prior steps.

    # 2. Execution
    cmd = [
        "python", "-m", "src.openrunyoro.make_chat",
        "--in", str(sample_input_path),
        "--out", str(output_jsonl_path),
        "--lang_model", "xx_sent_ud_sm"  # Using the model confirmed to be available
    ]
    
    # Set PYTHONPATH to include src directory
    # This ensures that 'import openrunyoro' works inside the script
    # when 'src' is a namespace package or similar structure.
    env = os.environ.copy()
    # Construct the path to the 'src' directory relative to the current working directory (project root)
    # Path.cwd() is typically the project root when running pytest
    python_path_src_dir = str(Path.cwd()) 
    
    # Add the constructed path to PYTHONPATH
    # This makes modules directly under 'src' (like 'openrunyoro') importable
    # as if 'src' itself is on the Python path.
    # Example: if openrunyoro is in src/openrunyoro, then `import openrunyoro`
    # would not work without src on path.
    # However, the script uses `python -m src.openrunyoro.make_chat`.
    # This -m flag changes how Python searches for modules.
    # It adds the current directory (project root) to sys.path.
    # So, `src.openrunyoro.make_chat` should be findable if `src` is a directory at project root
    # containing `openrunyoro` as a package.
    # The explicit PYTHONPATH manipulation might be redundant if pytest is run from project root.
    # Let's test without it first, as `-m` should handle it. If not, it can be added back.
    # env["PYTHONPATH"] = python_path_src_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)

    # 3. Assertions
    assert result.returncode == 0, f"make_chat.py failed. STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    assert output_jsonl_path.exists(), f"Output JSONL file was not created. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    conversations_count = 0
    total_turns_count = 0
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            conversations_count += 1
            assert line.strip().startswith("[") and line.strip().endswith("]"), \
                f"Line {line_idx+1} is not a JSON list: {line}"
            try:
                conversation = json.loads(line) 
            except json.JSONDecodeError as e:
                pytest.fail(f"Failed to parse JSON on line {line_idx+1}: {line}. Error: {e}")
            
            assert isinstance(conversation, list), f"Parsed JSON on line {line_idx+1} is not a list."
            assert len(conversation) > 0, f"Conversation on line {line_idx+1} is empty."
            assert len(conversation) <= 10, f"Conversation on line {line_idx+1} has more than 10 turns ({len(conversation)})."
            total_turns_count += len(conversation)
            for turn_idx, turn in enumerate(conversation):
                assert isinstance(turn, dict), f"Turn {turn_idx+1} in conversation {line_idx+1} is not a dictionary."
                assert "role" in turn, f"Turn {turn_idx+1} in conversation {line_idx+1} missing 'role'."
                assert "content" in turn, f"Turn {turn_idx+1} in conversation {line_idx+1} missing 'content'."
                assert turn["role"] in ["user", "assistant"], f"Invalid role in turn {turn_idx+1}, conversation {line_idx+1}: {turn['role']}"
                assert isinstance(turn["content"], str), f"Turn {turn_idx+1} content in conversation {line_idx+1} is not a string."
    
    # Recalculating expected turns from SAMPLE_TEXT_CONTENT:
    # Dialogue lines: "SpeakerA: Hello...", "SpeakerB: Greetings...", "SpeakerA: How are you...", 
    #                 "SpeakerB: I am doing...", "SpeakerA: I'm also good.", "SpeakerB: That is great...",
    #                 "SpeakerA: Let's talk...", "SpeakerB: Yes, let's.", "SpeakerA: I like...", "SpeakerB: Me too..."
    #                 = 10 dialogue lines -> 10 turns
    # Vocab lines: "omwana – child", "•enté – cow", "amasaka – sorghum", "igufa – bone", 
    #              "embwa – dog", "omwaka – year", "omutwe – head", "ekitabo – book"
    #              = 8 vocab lines -> 8 * 4 = 32 turns
    # Other lines: "This is a general statement...", "Another piece of information...", 
    #              "Yet another explanatory line.", "This could be a section on verbs.", "Final sentence for this test."
    #              = 5 other lines -> 5 * 2 = 10 turns
    # Total expected turns = 10 + 32 + 10 = 52 turns.
    # Expected conversations = ceil(52 / 10) = 6 conversations.
    
    assert total_turns_count == 52, f"Expected 52 total turns, got {total_turns_count}. Check segment processing. STDERR: {result.stderr}"
    assert conversations_count == 6, f"Expected 6 conversations, got {conversations_count}. STDERR: {result.stderr}"

    # Check logs
    log_file = Path("logs/make_chat.log")
    assert log_file.exists(), "Log file logs/make_chat.log was not created."
    
    log_content = log_file.read_text(encoding="utf-8")
    assert "Detected 10 dialogue segments." in log_content # Based on 10 speaker lines
    assert "Detected 8 vocabulary segments." in log_content # Based on 8 vocab lines
    assert "Detected 5 other segments." in log_content # Based on 5 other lines
    assert "Total chat turns generated: 52" in log_content
    assert "Grouped turns into 6 conversations." in log_content
    assert f"Writing 6 conversations to JSONL file: {str(output_jsonl_path)}" in log_content
    assert "Successfully written conversations to JSONL file." in log_content
    # The number of sentences found by Spacy can vary.
    # Example: "SpeakerA: Hello from Speaker A." is one sentence.
    # "omwana – child" is one sentence.
    # "This is a general statement about something interesting." is one sentence.
    # Original text has 23 lines with content. Most are single sentences.
    # "SpeakerB: I am doing well, A. And you?" -> Spacy might split this into two.
    # Let's aim for a rough estimate for sentences, e.g. > 20
    assert "Found" in log_content and "sentences after SpaCy processing." in log_content
    # A more precise sentence count would require running spacy locally on SAMPLE_TEXT_CONTENT.
    # For now, a general check should suffice.
    # Example: if spacy splits "SpeakerB: I am doing well, A. And you?" into two, then 10 dialogue lines might be 11-12 sentences.
    # 8 vocab lines, 5 other lines. Total ~24-25 sentences.
    # The test script currently logs "Detected X dialogue/vocab/other segments" based on matching these sentences.
    # This is what we are asserting above.
```
