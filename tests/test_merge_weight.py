import subprocess
import pytest
from pathlib import Path
import json
import os

# Sample data
PRIMARY_CONTENT = [
    [{"role": "user", "content": "Hello from primary A"}, {"role": "assistant", "content": "Primary Assistant A says hi"}],
    [{"role": "user", "content": "Primary Question B"}, {"role": "assistant", "content": "Primary Assistant B answers"}],
]
SECONDARY_CONTENT = [
    [{"role": "user", "content": "Hello from secondary C"}, {"role": "assistant", "content": "Secondary Assistant C says hi"}],
    [{"role": "user", "content": "Secondary Question D"}, {"role": "assistant", "content": "Secondary Assistant D answers"}],
    [{"role": "user", "content": "Nearly same as A"}, {"role": "assistant", "content": "Primary Assistant A says hi almost"}], # Near-duplicate of PRIMARY_CONTENT[0]
]

# Expected assistant text for the duplicate
DUPLICATE_ASSISTANT_TEXT = "Primary Assistant A says hi almost"

def write_jsonl(path, data_list_of_lists):
    with open(path, "w", encoding="utf-8") as f:
        for item_list in data_list_of_lists:
            f.write(json.dumps(item_list) + "\n")

def read_jsonl(path) -> list[list[dict]]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line.strip()))
    return lines

@pytest.fixture(scope="module", autouse=True)
def ensure_logs_dir_for_merge_weight():
    # Separate fixture name to avoid conflict if other tests use a similar one
    Path("logs").mkdir(exist_ok=True)

def test_merge_weight_basic(tmp_path):
    # 1. Setup
    primary_jsonl_path = tmp_path / "primary.jsonl"
    secondary_jsonl_path = tmp_path / "secondary.jsonl"
    train_jsonl_path = tmp_path / "train.jsonl"
    val_jsonl_path = tmp_path / "val.jsonl"

    write_jsonl(primary_jsonl_path, PRIMARY_CONTENT)
    write_jsonl(secondary_jsonl_path, SECONDARY_CONTENT)

    # 2. Execution
    cmd = [
        "python", "-m", "src.openrunyoro.merge_weight",
        "--primary", str(primary_jsonl_path),
        "--secondary", str(secondary_jsonl_path),
        "--weight", "3",
        "--val_split", "0.2", # For 8 items, 0.2 -> 1 val, 7 train
        "--out_train", str(train_jsonl_path),
        "--out_val", str(val_jsonl_path),
        "--seed", "123"
    ]
    env = os.environ.copy()
    # As per prompt, adding src to PYTHONPATH.
    # If project root is /app, and src is /app/src, this makes /app/src importable.
    # For `python -m src.openrunyoro.merge_weight`, this PYTHONPATH is usually not needed
    # if run from /app, as /app is added to sys.path by -m.
    # However, following the provided test structure.
    env["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + env.get("PYTHONPATH", "")
        
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)

    # 3. Assertions
    assert result.returncode == 0, f"merge_weight.py failed. STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    assert train_jsonl_path.exists(), "Train output file was not created."
    assert val_jsonl_path.exists(), "Validation output file was not created."

    train_conversations = read_jsonl(train_jsonl_path)
    val_conversations = read_jsonl(val_jsonl_path)

    # Expected counts:
    # Primary: 2 * 3 = 6
    # Secondary: 3
    # Total initial: 9
    # Duplicates removed: 1 (the one with "Primary Assistant A says hi almost")
    # Total after dedupe: 8
    # Val split 0.2 on 8 items: int(8 * 0.2) = 1. So 1 val, 7 train.
    expected_train_count = 7
    expected_val_count = 1
    
    assert len(train_conversations) == expected_train_count, f"Expected {expected_train_count} train conversations, got {len(train_conversations)}"
    assert len(val_conversations) == expected_val_count, f"Expected {expected_val_count} val conversations, got {len(val_conversations)}"
    
    all_output_conversations = train_conversations + val_conversations
    assert len(all_output_conversations) == 8, "Total output conversations mismatch"

    # Check for absence of the duplicate
    found_duplicate = False
    for conv_list in all_output_conversations:
        for turn in conv_list:
            if turn.get("role") == "assistant" and turn.get("content") == DUPLICATE_ASSISTANT_TEXT:
                found_duplicate = True
                break
        if found_duplicate:
            break
    assert not found_duplicate, "Near-duplicate conversation was found in the output."

    # Check stderr for class counts
    assert f"Training set size: {expected_train_count} conversations" in result.stderr
    assert f"Validation set size: {expected_val_count} conversations" in result.stderr
    
    # Check logs (simplified)
    log_file = Path("logs/merge_weight.log")
    assert log_file.exists(), "Log file 'logs/merge_weight.log' was not created."
    
    log_content = log_file.read_text(encoding="utf-8")
    assert f"Read {len(PRIMARY_CONTENT)} conversations from primary file" in log_content
    assert f"Read {len(SECONDARY_CONTENT)} conversations from secondary file" in log_content
    assert f"Primary data weighted by factor 3, resulting in {len(PRIMARY_CONTENT)*3} primary conversations" in log_content
    assert f"Combined data: {len(PRIMARY_CONTENT)*3} (weighted primary) + {len(SECONDARY_CONTENT)} (secondary) = {len(PRIMARY_CONTENT)*3 + len(SECONDARY_CONTENT)} total conversations before shuffling." in log_content
    assert f"Starting near-duplicate removal. Initial count: {len(PRIMARY_CONTENT)*3 + len(SECONDARY_CONTENT)}" in log_content
    # After removing 1 duplicate from 9 initial conversations
    assert f"Finished near-duplicate removal. Retained {len(PRIMARY_CONTENT)*3 + len(SECONDARY_CONTENT) - 1} conversations" in log_content
    assert f"Splitting data: {expected_train_count} for training, {expected_val_count} for validation." in log_content
    assert f"Training set size: {expected_train_count} conversations." in log_content # Note the period from the logger
    assert f"Validation set size: {expected_val_count} conversations." in log_content # Note the period from the logger
    assert f"Writing training set to: {str(train_jsonl_path)}" in log_content
    assert f"Writing validation set to: {str(val_jsonl_path)}" in log_content
    assert "Successfully wrote training set." in log_content
    assert "Successfully wrote validation set." in log_content
    assert "Processing complete." in log_content

```
