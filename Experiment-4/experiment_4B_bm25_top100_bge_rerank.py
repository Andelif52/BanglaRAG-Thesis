import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder


# ============================================================
# Paths
# ============================================================

CHUNKS_PATH = r"C:\Thesis\Andelif\bnwiki_chunks.jsonl"
QA_PATH = r"C:\Thesis\Andelif\qa_pairs_clean.json"

BM25_TOP200_CACHE_PATH = r"C:\Thesis\Andelif\Experiment-4\outputs\cache\bm25_top200_candidates_experiment_4A.jsonl"

OUTPUT_DIR = os.path.join("outputs", "experiment_4B_bm25_top100_bge_m3")

RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

for d in [OUTPUT_DIR, RESULTS_DIR, LOG_DIR, GRAPH_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# Settings
# ============================================================

METHOD_NAME = "BM25_Top100_plus_BGE_M3"
CANDIDATE_TOP_N = 100
TOP_K_VALUES = [1, 3, 5, 10, 20, 50, 100]

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 16

LOG_FILE = os.path.join(LOG_DIR, "experiment_4B_bm25_top100_bge_rerank_log.txt")


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
    chunk_map = {}

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

            chunks.append(obj)
            chunk_map[str(chunk_id)] = obj

    log(f"Loaded chunks: {len(chunks):,}")
    return chunks, chunk_map


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


def load_bm25_top100_cache(path, qa_pairs):
    log(f"Loading BM25 Top-{CANDIDATE_TOP_N} candidates from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BM25 Top-200 cache not found: {path}\n"
            f"Run Experiment 4A first."
        )

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            top_ids = obj.get("top_chunk_ids", [])
            top_scores = obj.get("top_scores", [])

            rows.append({
                "qa_index": obj.get("qa_index"),
                "qa_id": obj.get("qa_id"),
                "question": obj.get("question"),
                "gold_chunk_id": str(obj.get("gold_chunk_id")),
                "bm25_gold_found_rank": obj.get("gold_found_rank"),
                "top_chunk_ids": [str(x) for x in top_ids[:CANDIDATE_TOP_N]],
                "top_scores": top_scores[:CANDIDATE_TOP_N],
            })

    if len(rows) != len(qa_pairs):
        raise ValueError(
            f"BM25 cache row count mismatch. Cache rows: {len(rows)}, QA pairs: {len(qa_pairs)}"
        )

    log(f"Loaded BM25 Top-{CANDIDATE_TOP_N} candidates for {len(rows):,} queries.")
    return rows


# ============================================================
# Reranking
# ============================================================

def rerank_with_cross_encoder(model, qa_pairs, bm25_rows, chunk_map):
    log("=" * 80)
    log(f"Running {METHOD_NAME}")
    log(f"Reranker model: {RERANKER_MODEL_NAME}")
    log(f"Candidate top-N: {CANDIDATE_TOP_N}")
    log("=" * 80)

    all_results = []

    rerank_start = time.time()

    for qa, bm25_row in tqdm(
        zip(qa_pairs, bm25_rows),
        total=len(qa_pairs),
        desc=f"Reranking Top-{CANDIDATE_TOP_N}"
    ):
        question = qa["question"]
        gold_chunk_id = qa["gold_chunk_id"]
        candidate_ids = bm25_row["top_chunk_ids"]

        pairs = []

        for cid in candidate_ids:
            chunk = chunk_map.get(cid)
            if chunk is None:
                passage = ""
            else:
                passage = chunk["text"]

            pairs.append([question, passage])

        scores = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        scores = np.asarray(scores).reshape(-1)

        ranked_indices = np.argsort(scores)[::-1]

        reranked_ids = [candidate_ids[i] for i in ranked_indices]
        reranked_scores = [float(scores[i]) for i in ranked_indices]

        gold_found_rank = None
        if gold_chunk_id in reranked_ids:
            gold_found_rank = reranked_ids.index(gold_chunk_id) + 1

        all_results.append({
            "qa_index": qa["qa_index"],
            "qa_id": qa["qa_id"],
            "question": question,
            "answer": qa["answer"],
            "gold_chunk_id": gold_chunk_id,
            "bm25_gold_found_rank": bm25_row["bm25_gold_found_rank"],
            "reranked_gold_found_rank": gold_found_rank,
            "reranked_chunk_ids": reranked_ids,
            "reranked_scores": reranked_scores,
        })

    rerank_time = time.time() - rerank_start
    avg_rerank_latency_ms = (rerank_time / len(qa_pairs)) * 1000

    log(f"Reranking finished in {rerank_time:.2f} seconds.")
    log(f"Average rerank latency: {avg_rerank_latency_ms:.2f} ms/query.")

    return all_results, rerank_time, avg_rerank_latency_ms


# ============================================================
# Evaluation
# ============================================================

def evaluate_results(results, rerank_time_sec, avg_rerank_latency_ms):
    summary = {
        "method": METHOD_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "num_queries": len(results),
        "candidate_top_n": CANDIDATE_TOP_N,
        "rerank_time_sec": rerank_time_sec,
        "avg_rerank_latency_ms": avg_rerank_latency_ms,
    }

    for k in TOP_K_VALUES:
        hits = 0
        rr_sum = 0.0
        ndcg_sum = 0.0

        for row in results:
            rank = row["reranked_gold_found_rank"]

            if rank is not None and rank <= k:
                hits += 1
                rr_sum += 1.0 / rank
                ndcg_sum += 1.0 / np.log2(rank + 1)

        summary[f"recall@{k}"] = hits / len(results)
        summary[f"mrr@{k}"] = rr_sum / len(results)
        summary[f"ndcg@{k}"] = ndcg_sum / len(results)

    return summary


def build_detailed_dataframe(results):
    rows = []

    for row in results:
        out = {
            "method": METHOD_NAME,
            "qa_index": row["qa_index"],
            "qa_id": row["qa_id"],
            "question": row["question"],
            "answer": row["answer"],
            "gold_chunk_id": row["gold_chunk_id"],
            "bm25_gold_found_rank": row["bm25_gold_found_rank"],
            "reranked_gold_found_rank": row["reranked_gold_found_rank"],
        }

        for rank in range(1, CANDIDATE_TOP_N + 1):
            out[f"rank_{rank}_chunk_id"] = row["reranked_chunk_ids"][rank - 1]
            out[f"rank_{rank}_score"] = row["reranked_scores"][rank - 1]

        rows.append(out)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    experiment_start = time.time()

    log("=" * 80)
    log("Starting Experiment 4B: BM25 Top-100 + BGE-M3 Reranking")
    log("=" * 80)

    log(f"Torch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")

    chunks, chunk_map = load_chunks_jsonl(CHUNKS_PATH)
    qa_pairs = load_qa_pairs(QA_PATH)

    missing_gold = [
        qa["gold_chunk_id"]
        for qa in qa_pairs
        if qa["gold_chunk_id"] not in chunk_map
    ]

    if missing_gold:
        log(f"WARNING: Missing gold chunks: {len(missing_gold)}")
    else:
        log("All QA gold chunk IDs exist in the corpus.")

    bm25_rows = load_bm25_top100_cache(BM25_TOP200_CACHE_PATH, qa_pairs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading CrossEncoder on device: {device}")

    model = CrossEncoder(
        RERANKER_MODEL_NAME,
        device=device,
        max_length=512,
    )

    results, rerank_time_sec, avg_rerank_latency_ms = rerank_with_cross_encoder(
        model=model,
        qa_pairs=qa_pairs,
        bm25_rows=bm25_rows,
        chunk_map=chunk_map,
    )

    summary = evaluate_results(
        results=results,
        rerank_time_sec=rerank_time_sec,
        avg_rerank_latency_ms=avg_rerank_latency_ms,
    )

    summary_df = pd.DataFrame([summary])

    summary_path = os.path.join(
        RESULTS_DIR,
        "retrieval_summary_experiment_4B_bm25_top100_bge_m3.csv"
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    log(f"Saved summary: {summary_path}")

    detailed_df = build_detailed_dataframe(results)

    detailed_path = os.path.join(
        RESULTS_DIR,
        "detailed_query_results_experiment_4B_bm25_top100_bge_m3.csv"
    )
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    log(f"Saved detailed results: {detailed_path}")

    total_time = time.time() - experiment_start

    log("=" * 80)
    log("Experiment 4B finished.")
    log(f"Total experiment time: {total_time:.2f} seconds.")
    log("=" * 80)

    print("\nFinal Summary:")
    print(summary_df.to_string(index=False))

    print("\nImportant Values:")
    print(f"Recall@1  : {summary['recall@1']:.4f}")
    print(f"Recall@5  : {summary['recall@5']:.4f}")
    print(f"Recall@10 : {summary['recall@10']:.4f}")
    print(f"Recall@50 : {summary['recall@50']:.4f}")
    print(f"Recall@100: {summary['recall@100']:.4f}")
    print(f"MRR@10    : {summary['mrr@10']:.4f}")
    print(f"nDCG@10   : {summary['ndcg@10']:.4f}")
    print(f"Avg rerank latency: {summary['avg_rerank_latency_ms']:.2f} ms/query")


if __name__ == "__main__":
    main()