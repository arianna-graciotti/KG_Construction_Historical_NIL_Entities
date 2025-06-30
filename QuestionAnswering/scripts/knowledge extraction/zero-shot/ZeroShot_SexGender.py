import os
import logging
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from litellm import completion  # make sure this import is available

# ---------------- Configure Environment ----------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OPENROUTER_API_KEY"] = ""   # ← replace with your key

# ---------------- Reinitialize Logging ----------------
def init_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    fh = logging.FileHandler("/lookup_tables/QA_sexGender/ZS_answers/logs/general_log.log", mode="w")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    log = logging.getLogger(); log.setLevel(logging.INFO)
    log.addHandler(ch); log.addHandler(fh)
    log.info("Logger re-initialized.")
    return log

logger = init_logging()

# ---------------- CONFIG SECTION ----------------
CONFIG = {
    "INPUT_CSVS": [
        "/home/arianna/PycharmProjects/NIL_Grounding/lookup_tables/QA_sexGender/questions/QA_Gender_QID.csv",
        "/home/arianna/PycharmProjects/NIL_Grounding/lookup_tables/QA_sexGender/questions/QA_Gender_nil.csv"
    ],
    # pool of LLMs to test:
    "LLM_MODELS": [
        "openrouter/openai/gpt-4o-mini",
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "openrouter/qwen/qwen-2.5-72b-instruct",
        "openrouter/google/gemma-2-27b-it",
        "openrouter/microsoft/phi-3-medium-128k-instruct",
        "openrouter/mistralai/mixtral-8x7b-instruct",
    ],
    "LLM_MAX_TOKENS": 256,
    "ONLY_PROCESS_ONE_INPUT": False,
    "INPUT_CSV_INDEX": 0,
}

today_str = datetime.today().strftime('%Y%m%d')
OUTPUT_DIR = f"/lookup_tables/QA_sexGender/ZS_answers/ZS_{today_str}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Output directory: {OUTPUT_DIR}")

# ---------------- Core Functions ----------------
def query_llm(prompt: str, model: str) -> str:
    """
    Send `prompt` to the specified `model` and return its answer.
    """
    try:
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=CONFIG["LLM_MAX_TOKENS"]
        )
        if "choices" in resp and resp["choices"]:
            return resp["choices"][0]["message"]["content"].strip()
        return resp.get("response", "").strip()
    except Exception as e:
        logger.error(f"LLM ({model}) query error: {e}")
        return f"Error: {e}"

def check_llm_availability(model: str) -> bool:
    """
    Quick test on `model` to see if it responds without error.
    """
    test = query_llm("Hello!", model)
    if test.startswith("Error"):
        logger.warning(f"{model} unavailable, skipping.")
        return False
    logger.info(f"{model} is available.")
    return True

def process_without_retrieval(df: pd.DataFrame, csv_basename: str, model: str):
    """
    Loop through the DataFrame, send each row to `model` with no extra context,
    and write out a CSV.
    """
    logger.info(f"→ {csv_basename} on {model} (no context)")
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        # pull the needed fields out of the row
        span              = r["span"]
        original_sentence = r["sentence"]
        question          = r["question"]

        prompt = (
            "Your task is to determine the sex or gender of a given person occurring in a given sentence based on some provided relevant context. "
            "Your answer MUST consist solely of the sex or gender (e.g., \"male\") with no extra text. "
            "NOTE: The example below is for demonstration purposes only and should not influence your answer.\n\n"
            "Example (for demonstration only):\n"
            "Given the person \"Sir Thomas Gladstone\" occurring in the sentence \"The chair was occupied en the occasion by Sir Thomas Gladstone, Bart.\",\n"
            "answer the following question: \"What is the sex or gender of Sir Thomas Gladstone?\"\n"
            "Answer:male\n\n"
            "Now, given the following inputs:\n"
            f"Person: \"{span}\"\n"
            f"Sentence: \"{original_sentence}\"\n"
            f"Question: \"{question}\"\n\n"
            "Answer:"
        )

        answer = query_llm(prompt, model)
        rows.append({
            "question":           question,
            "gold_answer":        r.get("gold sexGender", ""),
            "sentence":           original_sentence,
            "span":               span,
            "prompt_sent_to_llm": prompt,
            "llm_answer":         answer
        })

    out = pd.DataFrame(rows)
    fname = (
        f"{csv_basename}_noctx_"
        f"{model.replace('/', '_')}_"
        f"{today_str}.csv"
    )
    out.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    logger.info(f"  • Saved → {fname}")


def main():
    t0 = time.time()
    logger = init_logging()

    inputs = CONFIG["INPUT_CSVS"]
    if CONFIG["ONLY_PROCESS_ONE_INPUT"]:
        idx = CONFIG["INPUT_CSV_INDEX"]
        inputs = [inputs[idx]]
        logger.info(f"Single-input mode: {inputs[0]}")

    for model in CONFIG["LLM_MODELS"]:
        if not check_llm_availability(model):
            continue

        for path in inputs:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                continue

            basename = os.path.splitext(os.path.basename(path))[0]
            process_without_retrieval(df, basename, model)

    logger.info(f"All done in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
