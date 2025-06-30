import os
import pickle
import spacy
import time
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------
# Set the folder path containing your .txt files.
INPUT_FOLDER = "/home/arianna/PycharmProjects/MHERCL_OIE/data/periodicals_mini_corpus"  # CHANGE THIS to your folder path.
# Set the output pickle file path.
OUTPUT_PICKLE = "/home/arianna/PycharmProjects/MHERCL_OIE/data/periodicals_mini_corpus/mini_corpus_sentences.pkl"

# When in test mode, process only the smallest 10 files (in increasing size).
TEST_MODE = True
TEST_FILES_COUNT = 10


# ---------------- End CONFIGURATION ----------------

def load_and_split_by_sentences(folder_path: str) -> dict:
    """
    Reads .txt files in `folder_path` and splits each document into sentences
    using spaCy's transformer-based model ("en_core_web_trf").

    In test mode, only the smallest TEST_FILES_COUNT files (by file size, in increasing order) are processed.

    Returns a dictionary where:
      - Keys are filenames.
      - Values are sub-dictionaries mapping sentence numbers (starting at 1)
        to the sentence text.
    """
    corpus_dict = {}

    # Get all .txt files in the folder.
    files = [fname for fname in os.listdir(folder_path) if fname.lower().endswith('.txt')]

    if not files:
        print("No .txt files found in the folder!")
        return corpus_dict

    if TEST_MODE:
        # Sort files by increasing file size.
        files = sorted(files, key=lambda fname: os.path.getsize(os.path.join(folder_path, fname)))
        files = files[:TEST_FILES_COUNT]
        print(f"TEST MODE enabled: Processing {len(files)} smallest files: {files}")
    else:
        print(f"Processing all {len(files)} files.")

    total_files = len(files)

    # Load spaCy's transformer-based model.
    nlp = spacy.load("en_core_web_trf")
    overall_start = time.time()

    pbar = tqdm(files, desc="Processing files", unit="file", dynamic_ncols=True)
    for idx, fname in enumerate(pbar):
        full_path = os.path.join(folder_path, fname)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            sentence_dict = {i + 1: sent for i, sent in enumerate(sentences)}
            corpus_dict[fname] = sentence_dict
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

    return corpus_dict


if __name__ == '__main__':
    print("Loading and splitting documents into sentences...")
    corpus = load_and_split_by_sentences(INPUT_FOLDER)
    print(f"Loaded {len(corpus)} files with sentence splitting.")

    with open(OUTPUT_PICKLE, "wb") as pf:
        pickle.dump(corpus, pf)
    print(f"Corpus saved to pickle file: {OUTPUT_PICKLE}")

import pickle

with open("/home/arianna/PycharmProjects/MHERCL_OIE/data/periodicals_mini_corpus/mini_corpus_sentences.pkl", "rb") as f:
    corpus = pickle.load(f)

# Print the loaded dictionary (optionally pretty-print a summary)
print("Loaded corpus:")
for fname, paragraphs in corpus.items():
    print(f"\nFilename: {fname}")
    for sent_num, sentence in paragraphs.items():
        print(f"  Sentence {sent_num}: {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
