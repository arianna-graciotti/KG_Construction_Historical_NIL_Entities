import os
import pickle
import re
import time
from tqdm import tqdm
from nltk.tokenize import blankline_tokenize

# ---------------- CONFIGURATION ----------------
INPUT_FOLDER = #periodicals full text can be shared on demand.
OUTPUT_PICKLE = ""

# When in test mode, process only the smallest TEST_FILES_COUNT files (by file size)
TEST_MODE = False
TEST_FILES_COUNT = 10


# ---------------- End CONFIGURATION ----------------

def load_and_split_by_paragraphs(folder_path: str) -> dict:
    """
    Reads all .txt files in 'folder_path', splits each document into paragraphs,
    and returns a dictionary structured as:
      { filename: { 1: paragraph1, 2: paragraph2, ... }, ... }
    It uses NLTK's blankline_tokenize as the primary method. If only one paragraph is returned,
    it falls back to splitting on one or more newlines.
    """
    corpus = {}

    # Get all .txt files in the folder.
    files = [fname for fname in os.listdir(folder_path) if fname.lower().endswith('.txt')]
    if TEST_MODE:
        files = sorted(files, key=lambda fname: os.path.getsize(os.path.join(folder_path, fname)))
        files = files[:TEST_FILES_COUNT]
        print(f"TEST MODE enabled: processing the {len(files)} smallest files: {files}")
    else:
        print(f"Processing all {len(files)} files.")

    total_files = len(files)
    overall_start = time.time()

    pbar = tqdm(files, desc="Processing files", unit="file", dynamic_ncols=True)
    for idx, fname in enumerate(pbar):
        full_path = os.path.join(folder_path, fname)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            # First, try using blankline_tokenize.
            paragraphs = blankline_tokenize(text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            # Fallback: if only one paragraph was obtained, try splitting by single newlines.
            if len(paragraphs) <= 1:
                paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
            # Build a sub-dictionary with keys as increasing integers (starting at 1).
            para_dict = {i + 1: para for i, para in enumerate(paragraphs)}
            corpus[fname] = para_dict
        except Exception as e:
            print(f"Error processing file {full_path}: {e}")

        elapsed = time.time() - overall_start
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (total_files - idx - 1) / rate if rate > 0 else 0
        pbar.set_postfix({
            "Elapsed": f"{elapsed:.2f}s",
            "ETA": f"{remaining:.2f}s",
            "Rate": f"{rate:.2f} f/s"
        })

    return corpus


if __name__ == '__main__':
    print("Loading and splitting documents into paragraphs...")
    corpus_dict = load_and_split_by_paragraphs(INPUT_FOLDER)
    print(f"Loaded and processed {len(corpus_dict)} files.")

    with open(OUTPUT_PICKLE, "wb") as pf:
        pickle.dump(corpus_dict, pf)
    print(f"Corpus saved to pickle file: {OUTPUT_PICKLE}")