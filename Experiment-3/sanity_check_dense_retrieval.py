import os
import json
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

CHUNKS_PATH = r"C:\Thesis\Andelif\bnwiki_chunks.jsonl"
QA_PATH = r"C:\Thesis\Andelif\qa_pairs_clean.json"

OUTPUT_DIR = "outputs_sanity_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

NUM_QA_SAMPLES = 50
NUM_RANDOM_NEGATIVE_CHUNKS = 5000

TOP_K_VALUES = [1, 3, 5, 10]

MODELS_TO_TEST = [
    {
        "name": "Dense_BGE_M3",
        "model_name": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "name": "Dense_multilingual_E5_base",
        "model_name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
]

BATCH_SIZE = 16


# ============================================================
# Utility Functions
# ============================================================

def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_chunks_jsonl(path):
    """
    Loads Bangla Wikipedia chunks.

    Expected fields may include:
    - chunk_id
    - text
    - title
    - doc_id

    The script tries to be flexible with field names.
    """
    chunks = []
    chunk_map = {}

    log(f"Loading chunks from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            chunk_id = (
                item.get("chunk_id")
                or item.get("id")
                or item.get("chunkId")
                or str(line_idx)
            )

            text = (
                item.get("text")
                or item.get("chunk_text")
                or item.get("content")
                or item.get("passage")
                or ""
            )

            title = item.get("title", "")
            doc_id = item.get("doc_id", item.get("document_id", ""))

            chunk_obj = {
                "chunk_id": str(chunk_id),
                "text": str(text),
                "title": str(title),
                "doc_id": str(doc_id),
            }

            chunks.append(chunk_obj)
            chunk_map[str(chunk_id)] = chunk_obj

    log(f"Loaded chunks: {len(chunks):,}")
    return chunks, chunk_map


def load_qa_pairs(path):
    """
    Loads QA pairs.

    Expected fields may include:
    - question
    - gold_chunk_id / chunk_id / relevant_chunk_id
    """
    log(f"Loading QA pairs from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # If the JSON has a wrapper key
        for key in ["data", "qa_pairs", "questions", "items"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    qa_pairs = []

    for idx, item in enumerate(data):
        question = (
            item.get("question")
            or item.get("query")
            or item.get("q")
            or ""
        )

        gold_chunk_id = (
            item.get("gold_chunk_id")
            or item.get("chunk_id")
            or item.get("relevant_chunk_id")
            or item.get("positive_chunk_id")
            or item.get("context_id")
            or ""
        )

        answer = item.get("answer", item.get("answers", ""))

        if question and gold_chunk_id:
            qa_pairs.append({
                "qa_id": item.get("qa_id", item.get("id", idx)),
                "question": str(question),
                "answer": str(answer),
                "gold_chunk_id": str(gold_chunk_id),
            })

    log(f"Loaded QA pairs with gold chunk id: {len(qa_pairs):,}")
    return qa_pairs


def normalize_embeddings(embeddings):
    embeddings = np.asarray(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def build_faiss_index(embeddings):
    """
    Uses cosine similarity through normalized vectors + inner product index.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def evaluate_retrieval(results, top_k_values):
    metrics = {}

    for k in top_k_values:
        hit_count = 0
        reciprocal_ranks = []

        for row in results:
            gold = row["gold_chunk_id"]
            retrieved_ids = row["retrieved_chunk_ids"][:k]

            if gold in retrieved_ids:
                hit_count += 1
                rank = retrieved_ids.index(gold) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        metrics[f"recall@{k}"] = hit_count / len(results)
        metrics[f"mrr@{k}"] = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return metrics


def make_small_test_corpus(chunks, chunk_map, qa_pairs):
    """
    Makes a small corpus that MUST include the gold chunks.

    This is important because if the gold chunks are not in the test corpus,
    dense retrieval cannot possibly retrieve them.
    """
    random.seed(RANDOM_SEED)

    valid_qa = [qa for qa in qa_pairs if qa["gold_chunk_id"] in chunk_map]

    if len(valid_qa) < NUM_QA_SAMPLES:
        sampled_qa = valid_qa
    else:
        sampled_qa = random.sample(valid_qa, NUM_QA_SAMPLES)

    gold_chunk_ids = set(qa["gold_chunk_id"] for qa in sampled_qa)

    gold_chunks = [chunk_map[cid] for cid in gold_chunk_ids]

    negative_pool = [
        chunk for chunk in chunks
        if chunk["chunk_id"] not in gold_chunk_ids and len(chunk["text"].strip()) > 20
    ]

    sampled_negatives = random.sample(
        negative_pool,
        min(NUM_RANDOM_NEGATIVE_CHUNKS, len(negative_pool))
    )

    test_corpus = gold_chunks + sampled_negatives
    random.shuffle(test_corpus)

    log(f"Sampled QA pairs: {len(sampled_qa)}")
    log(f"Gold chunks included in test corpus: {len(gold_chunks)}")
    log(f"Random negative chunks included: {len(sampled_negatives)}")
    log(f"Total sanity-check corpus size: {len(test_corpus)}")

    return sampled_qa, test_corpus


def run_dense_sanity_check(model_config, sampled_qa, test_corpus):
    method_name = model_config["name"]
    model_name = model_config["model_name"]
    query_prefix = model_config["query_prefix"]
    passage_prefix = model_config["passage_prefix"]

    log("=" * 70)
    log(f"Running dense sanity check: {method_name}")
    log(f"Model: {model_name}")
    log("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)

    chunk_ids = [chunk["chunk_id"] for chunk in test_corpus]
    chunk_texts = [
        passage_prefix + chunk["text"].replace("\n", " ").strip()
        for chunk in test_corpus
    ]

    questions = [
        query_prefix + qa["question"].replace("\n", " ").strip()
        for qa in sampled_qa
    ]

    # Encode corpus
    log("Encoding corpus...")
    start = time.time()

    corpus_embeddings = model.encode(
        chunk_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    corpus_embeddings = normalize_embeddings(corpus_embeddings)

    corpus_encode_time = time.time() - start
    log(f"Corpus encoding time: {corpus_encode_time:.2f} seconds")

    # Build FAISS index
    log("Building FAISS index...")
    start = time.time()

    index = build_faiss_index(corpus_embeddings)

    index_build_time = time.time() - start
    log(f"FAISS index build time: {index_build_time:.2f} seconds")

    # Encode queries
    log("Encoding queries...")
    start = time.time()

    query_embeddings = model.encode(
        questions,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    query_embeddings = normalize_embeddings(query_embeddings)

    query_encode_time = time.time() - start
    log(f"Query encoding time: {query_encode_time:.2f} seconds")

    # Search
    max_k = max(TOP_K_VALUES)
    log(f"Searching top-{max_k}...")
    start = time.time()

    scores, indices = index.search(query_embeddings, max_k)

    search_time = time.time() - start
    avg_search_latency_ms = (search_time / len(sampled_qa)) * 1000

    log(f"Total search time: {search_time:.2f} seconds")
    log(f"Average search latency: {avg_search_latency_ms:.2f} ms/query")

    # Prepare results
    results = []

    for i, qa in enumerate(sampled_qa):
        retrieved_indices = indices[i].tolist()
        retrieved_scores = scores[i].tolist()

        retrieved_chunk_ids = [chunk_ids[idx] for idx in retrieved_indices]

        top_texts = []
        for rank, idx in enumerate(retrieved_indices[:5], start=1):
            chunk = test_corpus[idx]
            text_preview = chunk["text"].replace("\n", " ").strip()
            text_preview = text_preview[:300]

            top_texts.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "score": retrieved_scores[rank - 1],
                "text_preview": text_preview,
            })

        gold_found_rank = None
        if qa["gold_chunk_id"] in retrieved_chunk_ids:
            gold_found_rank = retrieved_chunk_ids.index(qa["gold_chunk_id"]) + 1

        results.append({
            "method": method_name,
            "qa_id": qa["qa_id"],
            "question": qa["question"],
            "gold_chunk_id": qa["gold_chunk_id"],
            "gold_found_rank": gold_found_rank,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_scores": retrieved_scores,
            "top5_details": top_texts,
        })

    metrics = evaluate_retrieval(results, TOP_K_VALUES)

    metrics.update({
        "method": method_name,
        "model_name": model_name,
        "num_queries": len(sampled_qa),
        "corpus_size": len(test_corpus),
        "corpus_encode_time_sec": corpus_encode_time,
        "query_encode_time_sec": query_encode_time,
        "index_build_time_sec": index_build_time,
        "avg_search_latency_ms": avg_search_latency_ms,
    })

    log("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            log(f"{key}: {value:.4f}")
        else:
            log(f"{key}: {value}")

    # Save detailed results
    detailed_rows = []

    for row in results:
        top_ids = row["retrieved_chunk_ids"]
        top_scores = row["retrieved_scores"]

        flat_row = {
            "method": row["method"],
            "qa_id": row["qa_id"],
            "question": row["question"],
            "gold_chunk_id": row["gold_chunk_id"],
            "gold_found_rank": row["gold_found_rank"],
        }

        for rank in range(1, max_k + 1):
            flat_row[f"rank_{rank}_chunk_id"] = top_ids[rank - 1]
            flat_row[f"rank_{rank}_score"] = top_scores[rank - 1]

        for detail in row["top5_details"]:
            r = detail["rank"]
            flat_row[f"rank_{r}_text_preview"] = detail["text_preview"]

        detailed_rows.append(flat_row)

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = os.path.join(
        OUTPUT_DIR,
        f"sanity_detailed_results_{method_name}.csv"
    )
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    log(f"Detailed results saved: {detailed_path}")

    return metrics


# ============================================================
# Main
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    log("=" * 70)
    log("Dense Retrieval Sanity Check for Experiment 3")
    log("=" * 70)

    log(f"Python CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")

    chunks, chunk_map = load_chunks_jsonl(CHUNKS_PATH)
    qa_pairs = load_qa_pairs(QA_PATH)

    sampled_qa, test_corpus = make_small_test_corpus(
        chunks=chunks,
        chunk_map=chunk_map,
        qa_pairs=qa_pairs,
    )

    all_metrics = []

    for model_config in MODELS_TO_TEST:
        try:
            metrics = run_dense_sanity_check(
                model_config=model_config,
                sampled_qa=sampled_qa,
                test_corpus=test_corpus,
            )
            all_metrics.append(metrics)

        except Exception as e:
            log(f"ERROR while running {model_config['name']}: {repr(e)}")

            all_metrics.append({
                "method": model_config["name"],
                "model_name": model_config["model_name"],
                "error": repr(e),
            })

    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(OUTPUT_DIR, "sanity_summary_dense_retrieval.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    log("=" * 70)
    log("Sanity check finished.")
    log(f"Summary saved: {summary_path}")
    log("=" * 70)

    print("\n\nFinal Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()