import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PUNCT_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": '-',
    "\u2014": '-',
}

def normalize_text(text: str) -> str:
    for src, tgt in PUNCT_MAP.items():
        text = text.replace(src, tgt)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+\s*", "", line)
        if line:
            lines.append(line)
    normalized = " ".join(lines).lower()
    return normalized


def build_manifest(audio_dir_base: Path, text_dir_base: Path, audio_fileset_id: str,
                   text_fileset_id: str, manifest_path: Path, language_code: str = "") -> int:
    audio_dir = audio_dir_base / audio_fileset_id
    text_dir = text_dir_base / text_fileset_id
    entries = []
    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        return 0
    for audio_file in sorted(audio_dir.glob("*.*")):
        if audio_file.suffix.lower() not in {".wav", ".mp3"}:
            continue
        base = audio_file.stem
        parts = base.split("_")
        if len(parts) < 3:
            logger.warning(f"Unexpected filename format: {audio_file.name}")
            continue
        book_id, chapter = parts[-2], parts[-1]
        text_file = text_dir / f"{book_id}_{chapter}.txt"
        if not text_file.exists():
            logger.warning(f"Missing text for {audio_file.name}")
            continue
        transcript = normalize_text(text_file.read_text(encoding="utf-8"))
        entries.append({"id": base, "wav": str(audio_file.resolve()), "wrd": transcript})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for ent in entries:
            json.dump(ent, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Wrote {len(entries)} entries to {manifest_path}")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="Create SpeechBrain manifest from Bible Brain downloads")
    parser.add_argument("--audio_dir_base", type=Path, required=True)
    parser.add_argument("--text_dir_base", type=Path, required=True)
    parser.add_argument("--audio_fileset_id", required=True)
    parser.add_argument("--text_fileset_id", required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--language_code", default="nyo")
    args = parser.parse_args()

    build_manifest(args.audio_dir_base, args.text_dir_base, args.audio_fileset_id,
                   args.text_fileset_id, args.manifest_path, args.language_code)


if __name__ == "__main__":
    main()
