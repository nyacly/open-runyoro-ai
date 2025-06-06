import os
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
import time

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://4.dbt.io/api" # Using v4 as per Bible Brain documentation

def discover_filesets(api_key: str, language_codes: list = None, media_types: list = None, organization_id: str = None):
    """
    Discovers filesets from the Bible Brain API.

    Args:
        api_key: Your DBP API key.
        language_codes: A list of language codes (e.g., ["nyo", "ttj"]) to filter by.
        media_types: A list of media types (e.g., ["audio", "text_plain"]) to filter by.
                        "audio_drama" for dramatized audio, "audio" for non-dramatized.
        organization_id: Filter by a specific organization ID.

    Returns:
        A list of fileset objects matching the criteria.
    """
    if language_codes is None:
        language_codes = ["nyo", "ttj"] # Defaults for Runyoro/Rutooro
    if media_types is None:
        media_types = ["audio", "audio_drama"]

    all_filesets = []
    page = 1
    limit = 100 # Max allowed by API is 1000, but smaller pages are safer for stability

    logger.info(f"Discovering filesets for languages: {language_codes}, media types: {media_types}")

    while True:
        params = {
            "v": 4,
            "key": api_key,
            "limit": limit,
            "page": page,
        }
        # The API seems to prefer language_code (singular) for filtering by one language,
        # and language_id for comma-separated list. Let's try to be flexible.
        # However, testing shows 'language_code' with comma-separated values works.
        if language_codes:
            params["language_code"] = ",".join(language_codes)

        # Add media type filter if specified
        # The API uses 'media' for type filtering, and it can be comma-separated
        if media_types:
            params["media"] = ",".join(media_types)

        if organization_id:
            params["organization_id"] = organization_id

        try:
            response = requests.get(f"{BASE_URL}/bibles/filesets", params=params)
            response.raise_for_status()
            data = response.json()

            current_filesets = data.get("data", [])
            if not current_filesets:
                logger.info("No more filesets found on this page.")
                break

            all_filesets.extend(current_filesets)
            logger.info(f"Found {len(current_filesets)} filesets on page {page}. Total found so far: {len(all_filesets)}")

            # Check if this is the last page
            meta = data.get("meta", {}).get("pagination", {})
            if meta.get("current_page") >= meta.get("total_pages"):
                logger.info("Reached the last page of filesets.")
                break

            page += 1
            time.sleep(0.5) # Be respectful to the API

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if response is not None:
                logger.error(f"Response content: {response.text}")
            return [] # Return empty list on error
        except ValueError as e: # Handles JSON decoding errors
            logger.error(f"JSON decoding failed: {e}")
            logger.error(f"Response content: {response.text}")
            return []


    # Further client-side filtering if necessary, as API filtering can be broad
    filtered_filesets = []
    for fs in all_filesets:
        # Ensure it's one of the requested languages (API might return related languages)
        # fs['iso'] seems to be the language code, fs['set_type_code'] gives 'audio', 'audio_drama', 'text_plain'
        # fs['media_type'] is 'Audio' or 'Text'
        correct_lang = any(lc.lower() == fs.get('iso','').lower() for lc in language_codes)
        correct_type = any(mt.lower() in fs.get('set_type_code','').lower() for mt in media_types)

        if correct_lang and correct_type:
            filtered_filesets.append(fs)

    logger.info(f"Discovered {len(filtered_filesets)} filesets matching criteria after client-side filtering.")
    return filtered_filesets


def get_bible_books(api_key: str, fileset_id: str):
    """
    Retrieves a list of books for a given fileset_id (text or audio).
    This is useful for then fetching chapters.
    """
    params = {"v": 4, "key": api_key}
    try:
        # The endpoint structure for books seems to be /bibles/filesets/{fileset_id}/books
        # However, the primary way to get book/chapter structure is via /bibles/{bible_id}/book
        # This requires a bible_id, not just a fileset_id.
        # Let's try to get bible_id from the fileset_id first.

        fs_response = requests.get(f"{BASE_URL}/bibles/filesets/{fileset_id}", params=params)
        fs_response.raise_for_status()
        fs_data = fs_response.json().get("data", [])

        if not fs_data or not isinstance(fs_data, list) or not fs_data[0].get("bible_id"):
            # Fallback if bible_id is not directly in fileset details or if data is not as expected
            # Sometimes bible_id is part of the fileset_id itself (e.g., ENGESVN2DA contains ENGESV)
            # This is heuristic and might need refinement.
            # For now, we assume the fileset details will contain the bible_id.
            # If not, we might need to query /bibles first.
            logger.warning(f"Could not reliably determine bible_id from fileset {fileset_id} details. Response: {fs_data}")
            # Attempt to parse from fileset_id (common pattern: 6-char ID like NYOBSN)
            parsed_bible_id = fileset_id[:6]
            logger.info(f"Attempting to use parsed bible_id: {parsed_bible_id} from fileset_id: {fileset_id}")
            bible_id_to_use = parsed_bible_id
        else:
            bible_id_to_use = fs_data[0]["bible_id"]
            logger.info(f"Using bible_id: {bible_id_to_use} from fileset {fileset_id} details.")

        # Now fetch books for that bible_id
        # The API expects /bibles/{bible_id}/book for book listing.
        # The parameter `testament` can be OT or NT. Without it, it lists all.
        books_params = {"v": 4, "key": api_key, "bible_id": bible_id_to_use}
        books_response = requests.get(f"{BASE_URL}/bibles/{bible_id_to_use}/book", params=books_params)
        books_response.raise_for_status()
        books_data = books_response.json().get("data", [])

        logger.info(f"Found {len(books_data)} books for bible_id {bible_id_to_use} (associated with fileset {fileset_id}).")
        return books_data
    except requests.exceptions.RequestException as e:
        logger.error(f"API request for books failed: {e}")
        if 'books_response' in locals() and books_response is not None:
            logger.error(f"Response content: {books_response.text}")
        elif 'fs_response' in locals() and fs_response is not None:
            logger.error(f"Response content (fileset query): {fs_response.text}")
        return []
    except ValueError as e: # Handles JSON decoding errors
            logger.error(f"JSON decoding failed for books: {e}")
            if 'books_response' in locals() and books_response is not None:
                 logger.error(f"Response content: {books_response.text}")
            return []


def download_audio(api_key: str, fileset_id: str, out_dir: Path, book_ids: list = None, max_retries: int = 3):
    """
    Downloads audio chapters for a given fileset_id.

    Args:
        api_key: Your DBP API key.
        fileset_id: The fileset ID for the audio data.
        out_dir: The directory to save downloaded audio files.
        book_ids: Optional list of specific book IDs (e.g., ["MAT", "MRK"]) to download. Downloads all if None.
        max_retries: Maximum number of retries for a download.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting audio download for fileset {fileset_id} to {out_dir}")

    # First, get all books for this fileset to iterate through chapters
    # The /download endpoint requires bible_id, book_id, chapter, and fileset_id (as dam_id)

    # Get bible_id from fileset details
    fs_params = {"v": 4, "key": api_key}
    try:
        fs_response = requests.get(f"{BASE_URL}/bibles/filesets/{fileset_id}", params=fs_params)
        fs_response.raise_for_status()
        fs_data_list = fs_response.json().get("data", [])
        if not fs_data_list or not isinstance(fs_data_list, list) or not fs_data_list[0].get("bible_id"):
            logger.error(f"Could not determine bible_id from fileset {fileset_id}. Response: {fs_data_list}")
            # Fallback: try to parse from fileset_id (e.g. NYOBSN)
            bible_id_from_fileset = fileset_id[:6]
            logger.warning(f"Attempting to use parsed bible_id: {bible_id_from_fileset}")
            if not bible_id_from_fileset: # If fileset_id is too short or not standard
                 logger.error(f"Cannot proceed without bible_id for fileset {fileset_id}")
                 return False # Indicate failure
            bible_id = bible_id_from_fileset
        else:
            bible_id = fs_data_list[0]["bible_id"]
        logger.info(f"Determined bible_id as {bible_id} for fileset {fileset_id}")

    except requests.exceptions.RequestException as e:
        logger.error(f"API request for fileset details failed: {e}")
        if fs_response is not None: logger.error(f"Response content: {fs_response.text}")
        return False
    except ValueError as e:
        logger.error(f"JSON decoding failed for fileset details: {e}")
        if fs_response is not None: logger.error(f"Response content: {fs_response.text}")
        return False

    # Get book and chapter information
    # The /download endpoint itself doesn't list chapters, we need to get them from /bibles/filesets/{fileset_id}
    # then iterate. The path usually is /bibles/filesets/{fileset_id}/{book_id}/{chapter}
    # The /download access method is different: /download/request_url

    # Let's get all audio file paths available for this fileset
    # The API endpoint /bibles/filesets/{fileset_id} can list files if they are directly associated.
    # However, for bulk audio, it's usually by chapter.
    # The structure for audio files is often /bibles/filesets/{fileset_id}/{book_id}/{chapter}
    # e.g. https://4.dbt.io/api/bibles/filesets/NYOBSNN2DA/MAT/1?v=4&key=YOUR_KEY

    books_and_chapters = [] # List of tuples (book_id, chapter_number, path_to_audio_file_in_API)

    all_books_meta = get_bible_books(api_key, fileset_id)
    if not all_books_meta:
        logger.error(f"Could not retrieve book list for fileset {fileset_id}. Cannot download audio.")
        return False

    books_to_process = all_books_meta
    if book_ids:
        books_to_process = [b for b in all_books_meta if b.get("book_id") in book_ids]
        logger.info(f"Filtering download to specific books: {book_ids}. Found {len(books_to_process)} matching books.")

    if not books_to_process:
        logger.warning(f"No books found to process for fileset {fileset_id} (after filtering by book_ids if any).")
        return True # No specific books to download, so technically success.

    logger.info(f"Found {len(books_to_process)} books for fileset {fileset_id}. Iterating through chapters...")

    for book_meta in tqdm(books_to_process, desc="Processing books"):
        book_id = book_meta.get("book_id")
        # Chapters are usually integers from 1 up to book_meta.get("chapters") list length or max chapter.
        # The 'chapters' field in book_meta is a list of chapter numbers.
        # Example: "chapters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

        chapters_list = book_meta.get("chapters", [])
        if not chapters_list: # Fallback if 'chapters' is not a list of numbers
            try:
                num_chapters = int(book_meta.get("number_of_chapters", 0)) # Some API versions might have this
                if num_chapters > 0: chapters_list = list(range(1, num_chapters + 1))
            except ValueError:
                logger.warning(f"Could not determine chapters for book {book_id}. Skipping.")
                continue

        if not chapters_list:
            logger.warning(f"No chapters found for book {book_id} in fileset {fileset_id}. Skipping book.")
            continue

        logger.debug(f"Processing book {book_id} with chapters: {chapters_list}")

        for chapter_num in tqdm(chapters_list, desc=f"Book {book_id} Chapters", leave=False):
            # Construct the path to get chapter audio details (which includes download URLs)
            # Path format: /bibles/filesets/{fileset_id}/{book_id}/{chapter}
            # e.g., /bibles/filesets/NYOBSNN2DA/MAT/1
            chapter_audio_url = f"{BASE_URL}/bibles/filesets/{fileset_id}/{book_id}/{chapter_num}"
            params = {"v": 4, "key": api_key, "asset_id": bible_id} # asset_id seems to be required, often same as bible_id

            retry_count = 0
            success = False
            while retry_count < max_retries and not success:
                try:
                    response = requests.get(chapter_audio_url, params=params, timeout=30)
                    response.raise_for_status()
                    chapter_data = response.json().get("data", [])

                    if not chapter_data:
                        logger.warning(f"No audio data found for {book_id} chapter {chapter_num} in fileset {fileset_id}. Response: {response.text}")
                        success = True # Mark as success to avoid retrying this specific chapter if API confirms no data
                        continue

                    # chapter_data is a list of dicts, each representing an audio segment (usually one per chapter)
                    for audio_segment in chapter_data:
                        # The 'path' field in audio_segment is the direct download link
                        download_url = audio_segment.get("path")
                        if not download_url:
                            logger.warning(f"No download path for {book_id} ch {chapter_num} segment. Data: {audio_segment}")
                            continue

                        # Determine filename (e.g., NYOBSNN2DA_MAT_001.mp3)
                        # Use book_id, chapter_num (padded), and try to get extension from URL or assume mp3
                        file_name_base = f"{fileset_id}_{book_id}_{str(chapter_num).zfill(3)}"
                        # Try to infer extension from download_url
                        file_ext = Path(download_url).suffix or ".mp3"
                        if not file_ext.startswith("."): file_ext = "." + file_ext # ensure dot
                        # remove query params from extension if any
                        file_ext = file_ext.split("?")[0]

                        output_filename = out_dir / f"{file_name_base}{file_ext}"

                        if output_filename.exists():
                            logger.info(f"File {output_filename} already exists. Skipping download.")
                            success = True # Move to next file
                            continue

                        logger.info(f"Downloading {download_url} to {output_filename}")

                        # Perform the download with streaming and progress bar
                        audio_response = requests.get(download_url, stream=True, timeout=60)
                        audio_response.raise_for_status()
                        total_size = int(audio_response.headers.get('content-length', 0))

                        with open(output_filename, 'wb') as f, tqdm(
                            desc=output_filename.name,
                            total=total_size,
                            unit='iB',
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar:
                            for chunk in audio_response.iter_content(chunk_size=8192):
                                size = f.write(chunk)
                                bar.update(size)
                        logger.info(f"Successfully downloaded {output_filename}")
                        success = True
                        time.sleep(0.2) # Small delay after successful download

                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout downloading {book_id} chapter {chapter_num}. Retrying ({retry_count+1}/{max_retries})...")
                    retry_count += 1
                    time.sleep(5 * (retry_count + 1)) # Exponential backoff
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error downloading {book_id} chapter {chapter_num}: {e}. Content: {response.text if 'response' in locals() and response is not None else 'N/A'}")
                    retry_count += 1
                    if response is not None and response.status_code == 404:
                        logger.warning(f"Chapter {book_id} {chapter_num} not found (404). Skipping.")
                        success = True # Treat as "success" for this chapter to move on
                        break
                    logger.info(f"Retrying ({retry_count}/{max_retries})...")
                    time.sleep(5 * (retry_count + 1)) # Exponential backoff
                except Exception as e: # Catch any other unexpected errors during download/write
                    logger.error(f"An unexpected error occurred for {book_id} chapter {chapter_num}: {e}. Retrying ({retry_count+1}/{max_retries})...")
                    retry_count += 1
                    time.sleep(5 * (retry_count + 1))

            if not success:
                logger.error(f"Failed to download {book_id} chapter {chapter_num} after {max_retries} retries.")
                # Optionally, decide if this should halt all downloads or just skip this chapter

    logger.info(f"Finished audio download process for fileset {fileset_id}.")
    return True


def get_chapter_text(api_key: str, fileset_id: str, book_id: str, chapter_num: int, max_retries: int = 3):
    """
    Retrieves plain text for a specific book and chapter from a fileset.

    Args:
        api_key: Your DBP API key.
        fileset_id: The fileset ID for the text data (e.g., NYOBSN).
        book_id: The OSIS book ID (e.g., "MAT", "MRK", "LUK", "JHN").
        chapter_num: The chapter number.
        max_retries: Maximum number of retries.

    Returns:
        A string containing the chapter text, or None if an error occurs.
        The text is a single string with verses separated by newlines if possible.
    """
    # API endpoint for verse text: /bibles/filesets/{fileset_id}/{book_id}/{chapter_num}
    # This is the same endpoint structure as audio chapters, but we expect text output.
    # Need to ensure the fileset_id provided is for a text fileset.
    # Example: https://4.dbt.io/api/bibles/filesets/NYOTBTN/MAT/1?v=4&key=YOUR_KEY&verse_start=1&verse_end=176
    # The API returns JSON with verse objects. We need to concatenate them.

    params = {
        "v": 4,
        "key": api_key,
        # "asset_id": "dbp-prod", # This might be needed for some text filesets
        # "dialect_id": "RUNY", # Example, might not be needed
    }
    # Optional: To get all verses, you can omit verse_start and verse_end, or set a large range.
    # Some text filesets might require `asset_id` to be set, often to `dbp-prod` or `dbp-prod-text`.
    # Let's try without first.

    url = f"{BASE_URL}/bibles/filesets/{fileset_id}/{book_id}/{chapter_num}"
    logger.info(f"Fetching text for fileset {fileset_id}, book {book_id}, chapter {chapter_num}")

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            verses_data = response.json().get("data", [])

            if not verses_data:
                logger.warning(f"No text data found for {fileset_id} {book_id}:{chapter_num}. Response: {response.text}")
                # This could be a valid case (e.g. chapter doesn't exist or no text for it)
                return "" # Return empty string for no data

            # Concatenate verse texts
            chapter_content = []
            for verse in verses_data:
                # verse_text might have HTML tags, though for 'text_plain' it should be clean.
                # We should strip them if any, but for now assume it's plain.
                # The field is 'verse_text'.
                text = verse.get("verse_text", "").strip()
                if text:
                    # Prepend verse number for clarity, though not strictly required by ASR task.
                    # chapter_content.append(f"{verse.get('verse_start')} {text}")
                    chapter_content.append(text)

            full_chapter_text = "\n".join(chapter_content)
            logger.info(f"Successfully fetched text for {fileset_id} {book_id}:{chapter_num}. Length: {len(full_chapter_text)} chars.")
            return full_chapter_text

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching text for {fileset_id} {book_id}:{chapter_num}. Retrying ({retry_count+1}/{max_retries})...")
            retry_count += 1
            time.sleep(2 * (retry_count + 1))
        except requests.exceptions.RequestException as e:
            logger.error(f"API request for text failed ({fileset_id} {book_id}:{chapter_num}): {e}")
            if response is not None:
                logger.error(f"Response status: {response.status_code}, content: {response.text}")
            # If 404, it means the specific chapter/book might not exist for this fileset
            if response is not None and response.status_code == 404:
                logger.warning(f"Text for {fileset_id} {book_id}:{chapter_num} not found (404). Returning empty.")
                return "" # Return empty string for not found
            retry_count += 1
            time.sleep(2 * (retry_count + 1))
        except ValueError as e: # Handles JSON decoding errors
            logger.error(f"JSON decoding failed for text ({fileset_id} {book_id}:{chapter_num}): {e}")
            if response is not None:
                logger.error(f"Response content: {response.text}")
            # This is likely a non-recoverable error for this call
            return None # Indicate error

    logger.error(f"Failed to fetch text for {fileset_id} {book_id}:{chapter_num} after {max_retries} retries.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Download Bible data from Bible Brain (Digital Bible Platform).")
    parser.add_argument("--api_key", required=True, help="Your DBP API key.")
    parser.add_argument("--dest", type=Path, default=Path("data/bible"), help="Destination directory for downloaded data.")

    parser.add_argument("--language_codes", type=str, default="nyo,ttj", help="Comma-separated language codes (e.g., nyo,ttj).")
    parser.add_argument("--fileset_ids_audio", type=str, help="Comma-separated specific audio fileset IDs to download (e.g., NYOBSNN2DA). Overrides language/type discovery for audio.")
    parser.add_argument("--fileset_id_text", type=str, help="Specific text fileset ID to use for fetching text (e.g., NYOTBTN). Required if downloading text.")

    parser.add_argument("--skip_audio_download", action="store_true", help="Skip downloading audio files.")
    parser.add_argument("--skip_text_download", action="store_true", help="Skip downloading text files.")
    parser.add_argument("--book_ids", type=str, help="Comma-separated specific book IDs to download (e.g., MAT,MRK). Applies to both audio and text.")


    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    lang_codes = [lc.strip() for lc in args.language_codes.split(",")]
    book_ids_list = [b.strip().upper() for b in args.book_ids.split(",")] if args.book_ids else None

    # --- Audio Download ---
    if not args.skip_audio_download:
        logger.info("--- Starting Audio Download Phase ---")
        audio_fileset_ids_to_download = []
        if args.fileset_ids_audio:
            audio_fileset_ids_to_download = [f.strip() for f in args.fileset_ids_audio.split(",")]
            logger.info(f"Using specified audio fileset IDs: {audio_fileset_ids_to_download}")
        else:
            logger.info(f"Discovering audio filesets for languages: {lang_codes}")
            # Discover audio filesets (both dramatized and non-dramatized)
            audio_filesets = discover_filesets(args.api_key, language_codes=lang_codes, media_types=["audio", "audio_drama"])
            if audio_filesets:
                for fs in audio_filesets:
                    logger.info(f"Found audio fileset: ID={fs['id']}, Name={fs['name']}, Language={fs['iso']}, Type={fs['set_type_code']}")
                    audio_fileset_ids_to_download.append(fs['id'])
            else:
                logger.warning("No audio filesets discovered for the given languages.")

        if not audio_fileset_ids_to_download:
            logger.warning("No audio fileset IDs specified or discovered. Skipping audio download.")
        else:
            for fileset_id in audio_fileset_ids_to_download:
                audio_out_dir = args.dest / "audio" / fileset_id
                audio_out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading audio for fileset {fileset_id} to {audio_out_dir}")
                download_audio(args.api_key, fileset_id, audio_out_dir, book_ids=book_ids_list)
        logger.info("--- Finished Audio Download Phase ---")
    else:
        logger.info("Skipping audio download as per --skip_audio_download.")

    # --- Text Download ---
    if not args.skip_text_download:
        logger.info("--- Starting Text Download Phase ---")
        text_fileset_id_to_use = args.fileset_id_text

        if not text_fileset_id_to_use:
            logger.info(f"Discovering text filesets for languages: {lang_codes}")
            text_filesets = discover_filesets(args.api_key, language_codes=lang_codes, media_types=["text_plain", "text_format"]) # text_format for USX etc.
            if text_filesets:
                # Prefer 'text_plain' if available for simplicity
                plain_text_fs = next((fs for fs in text_filesets if fs.get('set_type_code') == 'text_plain'), None)
                if plain_text_fs:
                    text_fileset_id_to_use = plain_text_fs['id']
                    logger.info(f"Using discovered plain text fileset: ID={plain_text_fs['id']}, Name={plain_text_fs['name']}")
                elif text_filesets: # Fallback to first text fileset found
                    text_fileset_id_to_use = text_filesets[0]['id']
                    logger.warning(f"No 'text_plain' fileset found. Using first available text fileset: ID={text_filesets[0]['id']}, Name={text_filesets[0]['name']}, Type={text_filesets[0]['set_type_code']}")
                else:
                    logger.warning("No text filesets discovered for the given languages.")
            else:
                logger.warning("No text filesets discovered for the given languages.")

        if not text_fileset_id_to_use:
            logger.error("No text fileset ID specified or discovered. Cannot download text.")
        else:
            text_out_dir = args.dest / "text" / text_fileset_id_to_use
            text_out_dir.mkdir(parents=True, exist_ok=True)

            # Get all books for this text fileset
            books_meta = get_bible_books(args.api_key, text_fileset_id_to_use)
            if not books_meta:
                logger.error(f"Could not retrieve book list for text fileset {text_fileset_id_to_use}. Skipping text download.")
            else:
                books_to_process = books_meta
                if book_ids_list:
                    books_to_process = [b for b in books_meta if b.get("book_id") in book_ids_list]
                    logger.info(f"Filtering text download to specific books: {book_ids_list}. Found {len(books_to_process)} matching books.")

                if not books_to_process:
                    logger.warning(f"No books found to process for text fileset {text_fileset_id_to_use} (after filtering).")
                else:
                    all_texts = {} # Store as book_id_chapter -> text
                    logger.info(f"Fetching text for {len(books_to_process)} books from fileset {text_fileset_id_to_use}")
                    for book_meta in tqdm(books_to_process, desc="Fetching text per book"):
                        book_id = book_meta.get("book_id")
                        chapters_list = book_meta.get("chapters", [])
                        if not chapters_list: # Fallback
                             try: num_chapters = int(book_meta.get("number_of_chapters",0)); chapters_list = list(range(1,num_chapters+1))
                             except ValueError: logger.warning(f"No chapters for book {book_id}"); continue

                        for chapter_num in tqdm(chapters_list, desc=f"Book {book_id} Text Chapters", leave=False):
                            chapter_text = get_chapter_text(args.api_key, text_fileset_id_to_use, book_id, chapter_num)
                            if chapter_text is not None: # None indicates error, empty string is valid (no text for chapter)
                                # Save each chapter as a separate file for now, or one big file?
                                # For ASR manifest, we'll need per-audio-file text.
                                # Let's save it per chapter for now.
                                chapter_file = text_out_dir / f"{book_id}_{str(chapter_num).zfill(3)}.txt"
                                with open(chapter_file, 'w', encoding='utf-8') as f:
                                    f.write(chapter_text)
                                logger.debug(f"Saved text for {book_id} chapter {chapter_num} to {chapter_file}")
                            else:
                                logger.error(f"Failed to retrieve text for {book_id} chapter {chapter_num}.")
                    logger.info(f"Finished text download for fileset {text_fileset_id_to_use} to {text_out_dir}")
        logger.info("--- Finished Text Download Phase ---")
    else:
        logger.info("Skipping text download as per --skip_text_download.")

if __name__ == "__main__":
    # Example Usage (for testing directly, replace with your actual key):
    # export DBP_API_KEY="YOUR_KEY_HERE"
    # python -m runyoro_speech_ai.data_ingestion.download_bible_brain --api_key $DBP_API_KEY --dest temp_bible_data --language_codes nyo --fileset_ids_audio NYOBSNN2DA --fileset_id_text NYOTBTN --book_ids MAT,MRK
    # python -m runyoro_speech_ai.data_ingestion.download_bible_brain --api_key $DBP_API_KEY --dest temp_bible_data --language_codes nyo --skip_text_download
    # python -m runyoro_speech_ai.data_ingestion.download_bible_brain --api_key $DBP_API_KEY --dest temp_bible_data --language_codes nyo --skip_audio_download --fileset_id_text NYOTBTN

    main()
