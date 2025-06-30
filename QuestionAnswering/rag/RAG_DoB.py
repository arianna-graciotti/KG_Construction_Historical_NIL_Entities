import os
import pickle
import logging
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# ---------------- Configure Environment ----------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Set the external API key (if required by your hosted LLM)
os.environ["OPENROUTER_API_KEY"] = ""  # Replace with your actual API key

# ---------------- Reinitialize Logging ----------------
def init_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("../logs/general_log.log", mode="w")
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Logger re-initialized.")
    return logger

logger = init_logging()

# ---------------- CONFIG SECTION ----------------
CONFIG = {
    "INPUT_CSVS": [
        "/home/arianna/PycharmProjects/NIL_Grounding/lookup_tables/QA_DoB/questions/QA_DoB_QID.csv",
        "/home/arianna/PycharmProjects/NIL_Grounding/lookup_tables/QA_DoB/questions/QA_DoB_NIL.csv"
    ],
    "PICKLE_FILE": "/home/arianna/PycharmProjects/NIL_Grounding/lookup_tables/periodicals_mini_corpus/corpus_paragraphs.pkl",
    "TOP_K": 3,
    "LLM_MODEL": "meta-llama/llama-3.3-70b-instruct",  # This will be set inside the loop below.
    "LLM_MAX_TOKENS": 256,
    "EMBEDDING_BATCH_SIZE": 4,
    "USE_CPU_FOR_EMBEDDING": False,
    "ONLY_PROCESS_ONE_INPUT": False,
    "INPUT_CSV_INDEX": 0,
    "LLM_MODELS": [
        "openrouter/openai/gpt-4o-mini",
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "openrouter/qwen/qwen-2.5-72b-instruct",
        "openrouter/google/gemma-2-27b-it",
        "openrouter/microsoft/phi-3-medium-128k-instruct",
        "openrouter/mistralai/mixtral-8x7b-instruct",
    ]
}

# List of retriever types to test.
ALL_RETRIEVER_TYPES = [
    "gtr-xl", "bm25", "contriever", "gtr-large", "gtr-xl", "bge-large", "instructor-xl"
]

today_str = datetime.today().strftime('%Y%m%d')

# Directory for output CSVs.
OUTPUT_DIR = f"/home/arianna/PycharmProjects/NIL_Grounding/data/QA_DoB/answers/{today_str}_OpenRouter_R2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Output directory created (if not existing): {OUTPUT_DIR}")

# Test mode: if True, process only a 20250507_OpenRouter_R1_fixed number of questions per CSV.
TEST_MODE = False
TEST_QUESTIONS_COUNT = 3

# ---------------- End CONFIG SECTION ----------------

def load_or_encode_embeddings(model, documents: List[str], encoding_path: str, batch_size: int):
    """
    For SentenceTransformer-based models: Load precomputed embeddings or compute and _cache them.
    """
    if os.path.exists(encoding_path):
        logger.info(f"Found saved embeddings at {encoding_path}. Loading...")
        with open(encoding_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        logger.info("No saved embeddings found. Computing embeddings...")
        embeddings = model.encode(
            documents,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            truncation=True,
            max_length=512
        )
        with open(encoding_path, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(f"Computed and saved embeddings to {encoding_path}.")
    return embeddings

def load_or_encode_contriever_embeddings(tokenizer, model, documents: List[str], encoding_path: str):
    """
    For transformer-based contriever models: Load precomputed embeddings or compute and _cache them.
    """
    if os.path.exists(encoding_path):
        logger.info(f"Found saved contriever embeddings at {encoding_path}. Loading...")
        with open(encoding_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        logger.info("No saved contriever embeddings found. Computing embeddings...")
        embeddings = []
        for doc in tqdm(documents, desc="Computing contriever embeddings", leave=False):
            inputs = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                model_output = model(**inputs)
            pooled = mean_pooling(model_output, inputs["attention_mask"])
            embeddings.append(pooled.squeeze(0))
        embeddings = torch.stack(embeddings)
        with open(encoding_path, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(f"Computed and saved contriever embeddings to {encoding_path}.")
    return embeddings

def query_llm(prompt: str) -> str:
    """
    Send the prompt to the current LLM using the OpenRouter-compatible API.
    """
    from litellm import completion
    try:
        response = completion(
            model=CONFIG["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=CONFIG.get("LLM_MAX_TOKENS", 256)
        )
        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"].strip()
        else:
            answer = response.get("response", "").strip()
        logger.info("LLM returned an answer.")
    except Exception as e:
        logger.error(f"LLM query error: {e}")
        answer = f"Error: LLM query failed. {e}"
    return answer

def check_llm_availability(llm_model: str) -> bool:
    """
    Perform a test query with the given LLM model.
    Returns True if successful; else False.
    """
    original_model = CONFIG["LLM_MODEL"]
    CONFIG["LLM_MODEL"] = llm_model
    test_prompt = "Say hello!"
    result = query_llm(test_prompt)
    CONFIG["LLM_MODEL"] = original_model  # restore
    if result.startswith("Error:"):
        logger.warning(f"LLM model {llm_model} unavailable. Skipping.")
        return False
    logger.info(f"LLM model {llm_model} is available.")
    return True

def load_corpus_from_pickle(pickle_path: str) -> List[Tuple[str, str]]:
    """
    Load the corpus from a pickle file. Expect a dictionary-of-dictionaries,
    returning a list of tuples (paragraph_id, paragraph_text).
    """
    with open(pickle_path, "rb") as f:
        corpus_dict = pickle.load(f)
    corpus_list = []
    for fname, para_dict in corpus_dict.items():
        for para_num, para_text in para_dict.items():
            paragraph_id = f"{fname}_p{para_num}"
            corpus_list.append((paragraph_id, para_text))
    logger.info(f"Loaded {len(corpus_list)} paragraphs from {pickle_path}.")
    return corpus_list

# ---------------- Retriever Classes ----------------

class BaseRetriever:
    def retrieve(self, query: str, top_k: int):
        raise NotImplementedError()

class BM25Retriever(BaseRetriever):
    def __init__(self, corpus: List[Tuple[str, str]]):
        from rank_bm25 import BM25Okapi
        self.corpus = corpus
        self.documents = [doc[1] for doc in corpus]
        tokenized_docs = [text.split() for text in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, top_k: int):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1]
        top_indices = ranked_indices[:top_k]
        return [self.corpus[i] for i in top_indices]

class VectorRetriever(BaseRetriever):
    def __init__(self, corpus: List[Tuple[str, str]], model_name: str):
        self.corpus = corpus
        self.documents = [c[1] for c in corpus]
        logger.info(f"Loading embeddings model '{model_name}'...")
        device = "cpu" if CONFIG.get("USE_CPU_FOR_EMBEDDING", False) else "cuda"
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()
        encoding_dir = "/home/arianna/PycharmProjects/NIL_Grounding/data/periodicals_mini_corpus/embeddings"
        os.makedirs(encoding_dir, exist_ok=True)
        encoding_file = os.path.join(encoding_dir, f"corpus_encoding_{model_name.replace('/', '_')}.pkl")
        self.doc_embeddings = load_or_encode_embeddings(
            self.model, self.documents, encoding_file, CONFIG["EMBEDDING_BATCH_SIZE"]
        )

    def retrieve(self, query: str, top_k: int):
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            truncation=True,
            max_length=512
        )
        scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), self.doc_embeddings, dim=1
        )
        ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
        top_indices = ranked_indices[:top_k]
        return [(self.corpus[i][0], self.corpus[i][1]) for i in top_indices]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class OfficialContrieverRetriever(BaseRetriever):
    def __init__(self, corpus: List[Tuple[str, str]]):
        self.corpus = corpus
        self.documents = [doc[1] for doc in corpus]
        self.filenames = [doc[0] for doc in corpus]
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.model.eval()
        encoding_dir = "/home/arianna/PycharmProjects/NIL_Grounding/data/periodicals_mini_corpus/embeddings"
        os.makedirs(encoding_dir, exist_ok=True)
        encoding_file = os.path.join(encoding_dir, "corpus_encoding_contriever.pkl")
        self.doc_embeddings = load_or_encode_contriever_embeddings(
            self.tokenizer, self.model, self.documents, encoding_file
        )

    def retrieve(self, query: str, top_k: int):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            model_output = self.model(**inputs)
        query_embedding = mean_pooling(model_output, inputs["attention_mask"]).squeeze(0)
        scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), self.doc_embeddings, dim=1
        )
        ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
        top_indices = ranked_indices[:top_k]
        return [(self.filenames[i], self.documents[i]) for i in top_indices]

class PromptRetriever(BaseRetriever):
    def __init__(self, corpus: List[Tuple[str, str]], model_name: str = "allenai/promptriever"):
        self.corpus = corpus
        self.documents = [c[1] for c in corpus]
        logger.info(f"Loading Promptriever model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        encoding_dir = "/home/arianna/PycharmProjects/NIL_Grounding/data/periodicals_mini_corpus/embeddings"
        os.makedirs(encoding_dir, exist_ok=True)
        encoding_file = os.path.join(encoding_dir, f"corpus_encoding_{model_name.replace('/', '_')}.pkl")
        self.doc_embeddings = load_or_encode_embeddings(
            self.model, self.documents, encoding_file, CONFIG["EMBEDDING_BATCH_SIZE"]
        )

    def retrieve(self, query: str, top_k: int):
        q_emb = self.model.encode(query, convert_to_tensor=True)
        scores = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), self.doc_embeddings, dim=1
        )
        ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
        top_indices = ranked_indices[:top_k]
        return [(self.corpus[i][0], self.corpus[i][1]) for i in top_indices]

def get_retriever(retriever_type: str, corpus: List[Tuple[str, str]]) -> BaseRetriever:
    rt = retriever_type.lower()
    if rt == "bm25":
        return BM25Retriever(corpus)
    elif rt == "contriever":
        return OfficialContrieverRetriever(corpus)
    elif rt == "gtr-large":
        return VectorRetriever(corpus, "sentence-transformers/gtr-t5-large")
    elif rt == "gtr-xl":
        return VectorRetriever(corpus, "sentence-transformers/gtr-t5-xxl")
    elif rt == "bge-large":
        return VectorRetriever(corpus, "BAAI/bge-large-en")
    elif rt == "instructor-xl":
        return VectorRetriever(corpus, "hkunlp/instructor-xl")
    elif rt == "promptriever":
        return PromptRetriever(corpus)
    else:
        logger.warning(f"Unknown retriever '{retriever_type}', defaulting to BM25.")
        return BM25Retriever(corpus)

# ---------------- Main Processing Function ----------------
def process_retriever(retriever_type: str, df_in: pd.DataFrame, corpus: List[Tuple[str, str]], csv_basename: str):
    logger.info(f"Processing retriever '{retriever_type}' for {csv_basename}...")
    try:
        retriever = get_retriever(retriever_type, corpus)
        results = []
        # Columns: "sentence", "span", "question" (with "gold _occupation" optional)
        for idx, row in tqdm(df_in.iterrows(), total=len(df_in), desc=f"{retriever_type}"):
            question = row["question"]
            ans = row.get("gold DoB", "")
            original_sentence = row["sentence"]
            span = row["span"]
            # Inside process_retriever loop
            query_str = (
                f"Given the person {span} occurring in the sentence {original_sentence} "
                f"answer the following question: {question}"
            )
            top_docs = retriever.retrieve(query_str, CONFIG["TOP_K"])
            retrieved_ids = [td[0] for td in top_docs]
            retrieved_texts = [td[1] for td in top_docs]
            context_dict = {id_: txt for id_, txt in zip(retrieved_ids, retrieved_texts)}
            prompt_context = str(context_dict)
            complete_prompt = (
                "Your task is to determine the date of birth of a given person occurring in a given sentence based on some provided relevant context. "
                "Your answer MUST consist solely of a date of birth (e.g., \"1728-05-11T00:00:00Z\") with no extra text. "
                "NOTE: The example below is for demonstration purposes only and should not influence your answer.\n\n"
                "Example (for demonstration only):\n"
                "Given the person \"Gavinies\" occurring in the sentence \"Gavinies published three books of sonatas, and several concertos, which are very highly esteemed by connoisseurs.\",\n"
                "answer the following question: \"What is the date of birth of Gavinies?\"\n"
                "Answer:1728-05-11T00:00:00Z\n\n"
                "Now, given the following inputs:\n"
                f"Person: \"{span}\"\n"
                f"Sentence: \"{original_sentence}\"\n"
                f"Context:\n{prompt_context}\n"
                f"Question: \"{question}\"\n\n"
                "Answer:"
            )
            llm_answer = query_llm(complete_prompt)
            out_row = {
                "retriever": retriever_type,
                "question": question,
                "gold_answer": ans,
                "document": original_sentence,
                "span": span,
                "retrieved_context_ids": retrieved_ids,
                "retrieved_context_texts": retrieved_texts,
                "complete_prompt": complete_prompt,
                "llm_answer": llm_answer
            }
            results.append(out_row)
        llm_name = CONFIG["LLM_MODEL"].replace("/", "_")
        output_csv = os.path.join(OUTPUT_DIR, f"{csv_basename}_output_{retriever_type}_{llm_name}_{today_str}.csv")
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)
        logger.info(f"Finished {retriever_type} for {csv_basename}! Output saved to {output_csv}")
    except torch.cuda.OutOfMemoryError as oom:
        logger.error(f"CUDA OOM with {retriever_type} on {csv_basename}: {oom}")
        torch.cuda.empty_cache()
    except Exception as ex:
        logger.error(f"Error in {retriever_type} on {csv_basename}: {ex}")
    finally:
        if 'retriever' in locals():
            del retriever
        torch.cuda.empty_cache()
        import gc
        gc.collect()

def main():
    overall_start = time.time()
    logger = init_logging()
    corpus = load_corpus_from_pickle(CONFIG["PICKLE_FILE"])
    input_csvs = CONFIG.get("INPUT_CSVS", [])
    if not input_csvs:
        logger.error("No input CSV files specified.")
        return
    if CONFIG.get("ONLY_PROCESS_ONE_INPUT", False):
        idx = CONFIG.get("INPUT_CSV_INDEX", 0)
        input_csvs = [input_csvs[idx]]
        logger.info(f"Running on single input CSV: {input_csvs[0]}")
    for llm in CONFIG["LLM_MODELS"]:
        if not check_llm_availability(llm):
            continue
        CONFIG["LLM_MODEL"] = llm
        logger.info(f"Starting experiments with LLM model: {llm}")
        for csv_path in input_csvs:
            try:
                df_in = pd.read_csv(csv_path, sep=",")
            except Exception as e:
                logger.error(f"Failed to load CSV {csv_path}: {e}")
                continue
            if TEST_MODE:
                df_in = df_in.head(TEST_QUESTIONS_COUNT)
                logger.info(f"TEST MODE: processing {TEST_QUESTIONS_COUNT} questions in {csv_path}.")
            else:
                logger.info(f"Processing all {len(df_in)} questions in {csv_path}.")
            csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
            for retriever_type in ALL_RETRIEVER_TYPES:
                start_time_rt = time.time()
                process_retriever(retriever_type, df_in, corpus, csv_basename)
                end_time_rt = time.time()
                logger.info(f"Time for {retriever_type} on {csv_path} with {llm}: {end_time_rt - start_time_rt:.2f} sec.")
    overall_end = time.time()
    logger.info(f"Total processing time: {overall_end - overall_start:.2f} seconds.")

if __name__ == "__main__":
    main()
