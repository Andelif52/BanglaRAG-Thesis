import os
import json
import time
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

CHUNKS_PATH = r"C:\Thesis\Andelif\bnwiki_chunks.jsonl"
QA_PATH = r"C:\Thesis\Andelif\qa_pairs_clean.json"

BM25_PICKLE_PATHS = [
    r"C:\Thesis\Andelif\Experiment-2\outputs\indexes\bm25.pkl",
    r"C:\Thesis\Andelif\Experiment-1\outputs\indexes\bm25.pkl",
]

OUTPUT_DIR = "outputs"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for d in [OUTPUT_DIR, CACHE_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# Settings
# ============================================================

TOP_K_VALUES = [1, 3, 5, 10, 20, 50, 100, 200]
MAX_CANDIDATES = 200

LOG_FILE = os.path.join(LOG_DIR, "experiment_4A_bm25_candidate_recall_log.txt")


# ============================================================
# Logging
# ============================================================

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================
# Data loading
# ============================================================

def load_chunks_jsonl(path):
    chunks = []
    chunk_id_to_index = {}

    log(f"Loading chunks from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Loading chunks")):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            chunk_id = (
                item.get("chunk_id")
                or item.get("id")
                or item.get("chunkId")
                or str(idx)
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
            source = item.get("source", "")

            obj = {
                "chunk_id": str(chunk_id),
                "doc_id": str(doc_id),
                "title": str(title),
                "text": str(text),
                "source": str(source),
            }

            chunk_id_to_index[str(chunk_id)] = len(chunks)
            chunks.append(obj)

    log(f"Loaded chunks: {len(chunks):,}")
    return chunks, chunk_id_to_index


def load_qa_pairs(path):
    log(f"Loading QA pairs from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
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
                "qa_index": idx,
                "qa_id": item.get("qa_id", item.get("id", idx)),
                "question": str(question),
                "answer": str(answer),
                "gold_chunk_id": str(gold_chunk_id),
            })

    log(f"Loaded QA pairs: {len(qa_pairs):,}")
    return qa_pairs


def find_bm25_pickle():
    for path in BM25_PICKLE_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find bm25.pkl in Experiment-2 or Experiment-1 outputs."
    )


def simple_bangla_tokenize(text):
    return str(text).split()


# ============================================================
# Retrieval
# ============================================================

def generate_bm25_top200_candidates(bm25, chunks, qa_pairs):
    log(f"Generating BM25 top-{MAX_CANDIDATES} candidates for all queries...")

    all_results = []

    start = time.time()

    for qa in tqdm(qa_pairs, desc=f"BM25 top-{MAX_CANDIDATES} retrieval"):
        query_tokens = simple_bangla_tokenize(qa["question"])

        scores = bm25.get_scores(query_tokens)

        top_indices = np.argpartition(scores, -MAX_CANDIDATES)[-MAX_CANDIDATES:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        top_chunk_ids = [chunks[i]["chunk_id"] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]

        gold = qa["gold_chunk_id"]

        gold_rank = None
        if gold in top_chunk_ids:
            gold_rank = top_chunk_ids.index(gold) + 1

        all_results.append({
            "qa_index": qa["qa_index"],
            "qa_id": qa["qa_id"],
            "question": qa["question"],
            "answer": qa["answer"],
            "gold_chunk_id": gold,
            "gold_found_rank": gold_rank,
            "top_chunk_ids": top_chunk_ids,
            "top_scores": top_scores,
        })

    elapsed = time.time() - start
    avg_latency_ms = (elapsed / len(qa_pairs)) * 1000

    log(f"BM25 top-{MAX_CANDIDATES} generation finished in {elapsed:.2f} seconds.")
    log(f"Average BM25 retrieval latency: {avg_latency_ms:.2f} ms/query.")

    return all_results, elapsed, avg_latency_ms


# ============================================================
# Evaluation
# ============================================================

def evaluate_candidate_recall(results, retrieval_time_sec, avg_latency_ms):
    summary = {
        "method": "BM25_candidate_pool_analysis",
        "num_queries": len(results),
        "max_candidates": MAX_CANDIDATES,
        "retrieval_time_sec": retrieval_time_sec,
        "avg_retrieval_latency_ms": avg_latency_ms,
    }

    for k in TOP_K_VALUES:
        hits = 0
        rr_sum = 0.0
        ndcg_sum = 0.0

        for row in results:
            gold_rank = row["gold_found_rank"]

            if gold_rank is not None and gold_rank <= k:
                hits += 1
                rr_sum += 1.0 / gold_rank
                ndcg_sum += 1.0 / np.log2(gold_rank + 1)
            else:
                rr_sum += 0.0
                ndcg_sum += 0.0

        summary[f"candidate_recall@{k}"] = hits / len(results)
        summary[f"mrr@{k}"] = rr_sum / len(results)
        summary[f"ndcg@{k}"] = ndcg_sum / len(results)

    return summary


def build_detailed_dataframe(results):
    rows = []

    for row in results:
        out = {
            "qa_index": row["qa_index"],
            "qa_id": row["qa_id"],
            "question": row["question"],
            "answer": row["answer"],
            "gold_chunk_id": row["gold_chunk_id"],
            "gold_found_rank": row["gold_found_rank"],
            "hit@50": row["gold_found_rank"] is not None and row["gold_found_rank"] <= 50,
            "hit@100": row["gold_found_rank"] is not None and row["gold_found_rank"] <= 100,
            "hit@200": row["gold_found_rank"] is not None and row["gold_found_rank"] <= 200,
        }

        for rank in range(1, MAX_CANDIDATES + 1):
            out[f"rank_{rank}_chunk_id"] = row["top_chunk_ids"][rank - 1]
            out[f"rank_{rank}_score"] = row["top_scores"][rank - 1]

        rows.append(out)

    return pd.DataFrame(rows)


def save_top_candidate_cache(results):
    cache_path = os.path.join(CACHE_DIR, "bm25_top200_candidates_experiment_4A.jsonl")

    with open(cache_path, "w", encoding="utf-8") as f:
        for row in results:
            obj = {
                "qa_index": row["qa_index"],
                "qa_id": row["qa_id"],
                "question": row["question"],
                "gold_chunk_id": row["gold_chunk_id"],
                "gold_found_rank": row["gold_found_rank"],
                "top_chunk_ids": row["top_chunk_ids"],
                "top_scores": row["top_scores"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    log(f"Saved BM25 top-200 candidate cache: {cache_path}")


# ============================================================
# Main
# ============================================================

def main():
    experiment_start = time.time()

    log("=" * 80)
    log("Starting Experiment 4A: BM25 Candidate Pool Recall Analysis")
    log("=" * 80)

    chunks, chunk_id_to_index = load_chunks_jsonl(CHUNKS_PATH)
    qa_pairs = load_qa_pairs(QA_PATH)

    missing_gold = [
        qa["gold_chunk_id"]
        for qa in qa_pairs
        if qa["gold_chunk_id"] not in chunk_id_to_index
    ]

    if missing_gold:
        log(f"WARNING: Missing gold chunks: {len(missing_gold)}")
    else:
        log("All QA gold chunk IDs exist in the corpus.")

    bm25_path = find_bm25_pickle()
    log(f"Loading BM25 index from: {bm25_path}")

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    results, retrieval_time_sec, avg_latency_ms = generate_bm25_top200_candidates(
        bm25=bm25,
        chunks=chunks,
        qa_pairs=qa_pairs,
    )

    summary = evaluate_candidate_recall(
        results=results,
        retrieval_time_sec=retrieval_time_sec,
        avg_latency_ms=avg_latency_ms,
    )

    summary_df = pd.DataFrame([summary])

    summary_path = os.path.join(
        RESULTS_DIR,
        "retrieval_summary_experiment_4A_bm25_candidate_pool.csv"
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    log(f"Saved summary: {summary_path}")

    detailed_df = build_detailed_dataframe(results)

    detailed_path = os.path.join(
        RESULTS_DIR,
        "detailed_query_results_experiment_4A_bm25_top200.csv"
    )
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    log(f"Saved detailed results: {detailed_path}")

    save_top_candidate_cache(results)

    total_time = time.time() - experiment_start

    log("=" * 80)
    log("Experiment 4A finished.")
    log(f"Total experiment time: {total_time:.2f} seconds.")
    log("=" * 80)

    print("\nFinal Summary:")
    print(summary_df.to_string(index=False))

    print("\nImportant Candidate Recall Values:")
    print(f"Candidate Recall@50  : {summary['candidate_recall@50']:.4f}")
    print(f"Candidate Recall@100 : {summary['candidate_recall@100']:.4f}")
    print(f"Candidate Recall@200 : {summary['candidate_recall@200']:.4f}")


if __name__ == "__main__":
    main()