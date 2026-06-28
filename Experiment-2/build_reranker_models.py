"""
Experiment 2: BM25 Top-50 Retrieval + Reranking
================================================

Purpose:
    1. Retrieve top-50 candidate chunks using BM25 over the full Bangla Wikipedia corpus.
    2. Rerank only those 50 candidates using multilingual rerankers.
    3. Evaluate Recall@K / Hit@K, MRR@K, nDCG@K, and latency.
    4. Generate thesis-ready CSV tables and graphs.

Recommended folder structure:

    Thesis/
    ├── bnwiki_chunks.jsonl
    ├── qa_pairs_clean.json
    ├── Experiment-1/
    │   └── outputs/
    └── Experiment-2/
        └── build_reranker_models.py

Run from inside Experiment-2:

    python build_reranker_models.py

Required installation:

    pip install rank_bm25 sentence-transformers transformers torch pandas numpy matplotlib tqdm accelerate

Notes:
    - This script does NOT embed all 481k chunks.
    - This script does NOT rerank the full corpus.
    - BM25 still searches the full corpus.
    - The rerankers only score top-50 BM25 candidates per query.
    - Candidate generation is cached, so if the script stops, rerunning will reuse BM25 candidates.
"""

import os
import gc
import json
import time
import pickle
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rank_bm25 import BM25Okapi

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ============================================================
# Configuration
# ============================================================

# Recommended: keep datasets in the project root and this script inside Experiment-2.
# These paths work when running from Experiment-2/.
CHUNKS_PATH = "../bnwiki_chunks.jsonl"
QA_PATH = "../qa_pairs_clean.json"

# Fallback: if the files are copied into Experiment-2 directly, these will be used.
FALLBACK_CHUNKS_PATH = "bnwiki_chunks.jsonl"
FALLBACK_QA_PATH = "qa_pairs_clean.json"

OUTPUT_DIR = Path("outputs")
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
INDEX_DIR = OUTPUT_DIR / "indexes"
LOG_DIR = OUTPUT_DIR / "logs"
CACHE_DIR = OUTPUT_DIR / "cache"

for directory in [RESULTS_DIR, FIGURES_DIR, INDEX_DIR, LOG_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TOP_N_CANDIDATES = 50
EVAL_K_VALUES = [1, 3, 5, 10, 50]

# Keep None for full experiment.
# Use a small number such as 100 only for debugging.
MAX_QUERIES = None

# Reranker batch size:
# - CPU: 4 or 8 is safer.
# - GPU: 16 or 32 can be faster depending on VRAM.
BATCH_SIZE_CPU = 4
BATCH_SIZE_GPU = 16

# Keep both for final experiment.
# If BGE is too slow or memory-heavy on your machine, set RUN_BGE_RERANKER = False.
RUN_MMARCO_RERANKER = True
RUN_BGE_RERANKER = True

RERANKERS = []
if RUN_MMARCO_RERANKER:
    RERANKERS.append({
        "method_name": "BM25_Top50_plus_mMARCO_MiniLM",
        "model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "description": "BM25 Top-50 candidates reranked using multilingual mMARCO MiniLM cross-encoder."
    })

if RUN_BGE_RERANKER:
    RERANKERS.append({
        "method_name": "BM25_Top50_plus_BGE_M3",
        "model_name": "BAAI/bge-reranker-v2-m3",
        "description": "BM25 Top-50 candidates reranked using BGE multilingual reranker."
    })

# Optional: reuse BM25 index from Experiment-1 if it exists.
EXPERIMENT1_BM25_CACHE = Path("../Experiment-1/outputs/indexes/bm25.pkl")
LOCAL_BM25_CACHE = INDEX_DIR / "bm25.pkl"

CANDIDATE_CACHE_PATH = CACHE_DIR / f"bm25_top{TOP_N_CANDIDATES}_candidates.jsonl"

# Optional: include Experiment-1 BM25 row in a comparison CSV/graph if available.
EXPERIMENT1_SUMMARY_PATH = Path("../Experiment-1/outputs/results/retrieval_summary.csv")


# ============================================================
# Logging
# ============================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with open(LOG_DIR / "experiment_2_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================
# Loading
# ============================================================

def resolve_path(primary: str, fallback: str) -> Path:
    primary_path = Path(primary)
    fallback_path = Path(fallback)

    if primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Could not find required file.\n"
        f"Tried: {primary_path.resolve()}\n"
        f"Also tried: {fallback_path.resolve()}\n"
    )


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}")
    return rows


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    return text.split()


# ============================================================
# Metrics
# ============================================================

def reciprocal_rank(retrieved_ids, relevant_id, k):
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id == relevant_id:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_id, k):
    # One known relevant chunk per question. Ideal DCG = 1.
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id == relevant_id:
            return 1.0 / np.log2(rank + 1)
    return 0.0


def add_metrics_to_row(row, retrieved_ids, relevant_id):
    for k in EVAL_K_VALUES:
        top_ids = retrieved_ids[:k]
        row[f"hit@{k}"] = 1 if relevant_id in top_ids else 0
        row[f"recall@{k}"] = row[f"hit@{k}"]
        row[f"mrr@{k}"] = reciprocal_rank(retrieved_ids, relevant_id, k)
        row[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_id, k)
    return row


def summarize_detailed_results(method_name, detailed_df):
    summary = {
        "method": method_name,
        "num_queries": int(len(detailed_df)),
        "avg_bm25_candidate_latency_ms": float(detailed_df["bm25_candidate_latency_ms"].mean()),
        "avg_rerank_latency_ms": float(detailed_df["rerank_latency_ms"].mean()),
        "avg_total_latency_ms": float(detailed_df["total_latency_ms"].mean()),
        "median_total_latency_ms": float(detailed_df["total_latency_ms"].median()),
        "max_total_latency_ms": float(detailed_df["total_latency_ms"].max()),
    }

    for k in EVAL_K_VALUES:
        summary[f"hit@{k}"] = float(detailed_df[f"hit@{k}"].mean())
        summary[f"recall@{k}"] = float(detailed_df[f"recall@{k}"].mean())
        summary[f"mrr@{k}"] = float(detailed_df[f"mrr@{k}"].mean())
        summary[f"ndcg@{k}"] = float(detailed_df[f"ndcg@{k}"].mean())

    return summary


# ============================================================
# Dataset statistics
# ============================================================

def save_dataset_statistics(chunks, qa_pairs):
    chunk_lengths = [len(c.get("text", "").split()) for c in chunks]
    qa_chunk_counts = pd.Series([qa["chunk_id"] for qa in qa_pairs]).value_counts()

    stats = {
        "total_chunks": len(chunks),
        "total_qa_pairs": len(qa_pairs),
        "unique_docs": len(set(c.get("doc_id", "") for c in chunks)),
        "unique_qa_chunks": len(set(qa["chunk_id"] for qa in qa_pairs)),
        "avg_chunk_words": float(np.mean(chunk_lengths)),
        "median_chunk_words": float(np.median(chunk_lengths)),
        "min_chunk_words": int(np.min(chunk_lengths)),
        "max_chunk_words": int(np.max(chunk_lengths)),
        "avg_qa_per_relevant_chunk": float(qa_chunk_counts.mean()),
        "max_qa_per_relevant_chunk": int(qa_chunk_counts.max()),
        "top_n_candidates_for_reranking": TOP_N_CANDIDATES,
    }

    pd.DataFrame([stats]).to_csv(
        RESULTS_DIR / "dataset_statistics_experiment_2.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return stats


# ============================================================
# BM25
# ============================================================

def load_or_build_bm25(chunks):
    if LOCAL_BM25_CACHE.exists():
        log(f"Loading local BM25 cache from {LOCAL_BM25_CACHE}")
        with open(LOCAL_BM25_CACHE, "rb") as f:
            return pickle.load(f), 0.0, "loaded_local_cache"

    if EXPERIMENT1_BM25_CACHE.exists():
        log(f"Loading BM25 cache from Experiment-1: {EXPERIMENT1_BM25_CACHE}")
        with open(EXPERIMENT1_BM25_CACHE, "rb") as f:
            bm25 = pickle.load(f)

        # Save a local copy for Experiment-2 reproducibility.
        with open(LOCAL_BM25_CACHE, "wb") as f:
            pickle.dump(bm25, f)

        return bm25, 0.0, "loaded_experiment_1_cache"

    log("BM25 cache not found. Building BM25 index from chunks...")
    start = time.perf_counter()
    tokenized_corpus = [simple_tokenize(c.get("text", "")) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    build_time = time.perf_counter() - start

    with open(LOCAL_BM25_CACHE, "wb") as f:
        pickle.dump(bm25, f)

    log(f"BM25 index built in {build_time:.2f} seconds and saved to {LOCAL_BM25_CACHE}")
    return bm25, build_time, "built_new"


def get_top_n_indices(scores, n):
    """Fast top-N selection using argpartition, then exact sort only among top-N."""
    scores = np.asarray(scores)
    if n >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        candidate_indices = np.argpartition(scores, -n)[-n:]
        top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    return top_indices


def generate_or_load_bm25_candidates(bm25, chunks, qa_pairs):
    expected_count = len(qa_pairs)

    if CANDIDATE_CACHE_PATH.exists():
        log(f"Found BM25 candidate cache: {CANDIDATE_CACHE_PATH}")
        cached_rows = []
        with open(CANDIDATE_CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cached_rows.append(json.loads(line))

        if len(cached_rows) == expected_count:
            log(f"Using cached BM25 top-{TOP_N_CANDIDATES} candidates for {len(cached_rows)} queries.")
            return cached_rows

        log(
            f"Candidate cache has {len(cached_rows)} rows but expected {expected_count}. "
            f"Regenerating cache."
        )

    log(f"Generating BM25 top-{TOP_N_CANDIDATES} candidates for all queries...")
    candidate_rows = []

    iterator = enumerate(qa_pairs)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(qa_pairs), desc="BM25 candidate generation")

    with open(CANDIDATE_CACHE_PATH, "w", encoding="utf-8") as f:
        for idx, qa in iterator:
            question = qa["question"]
            relevant_id = qa["chunk_id"]

            start = time.perf_counter()
            scores = bm25.get_scores(simple_tokenize(question))
            top_indices = get_top_n_indices(scores, TOP_N_CANDIDATES)
            latency = time.perf_counter() - start

            candidate_ids = [chunks[int(i)]["chunk_id"] for i in top_indices]
            candidate_scores = [float(scores[int(i)]) for i in top_indices]

            row = {
                "query_index": idx,
                "question": question,
                "answer": qa.get("answer", ""),
                "relevant_chunk_id": relevant_id,
                "candidate_indices": [int(i) for i in top_indices],
                "candidate_chunk_ids": candidate_ids,
                "candidate_bm25_scores": candidate_scores,
                "bm25_candidate_latency_ms": latency * 1000,
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            candidate_rows.append(row)

    log(f"BM25 candidate cache saved to {CANDIDATE_CACHE_PATH}")
    return candidate_rows


def evaluate_bm25_top50_candidate_order(candidate_rows):
    log("Evaluating BM25 Top-50 candidate order as baseline for Experiment-2...")

    rows = []
    for item in candidate_rows:
        retrieved_ids = item["candidate_chunk_ids"]
        relevant_id = item["relevant_chunk_id"]

        row = {
            "method": "BM25_Top50_candidate_order",
            "query_index": item["query_index"],
            "question": item["question"],
            "answer": item.get("answer", ""),
            "relevant_chunk_id": relevant_id,
            "retrieved_chunk_ids": json.dumps(retrieved_ids, ensure_ascii=False),
            "bm25_candidate_latency_ms": item["bm25_candidate_latency_ms"],
            "rerank_latency_ms": 0.0,
            "total_latency_ms": item["bm25_candidate_latency_ms"],
        }

        row = add_metrics_to_row(row, retrieved_ids, relevant_id)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(
        RESULTS_DIR / "detailed_query_results_BM25_Top50_candidate_order.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return summarize_detailed_results("BM25_Top50_candidate_order", df), df


# ============================================================
# Reranker
# ============================================================

class PairReranker:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        log(f"Loading reranker model: {model_name}")
        log(f"Using device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, pairs, batch_size):
        all_scores = []

        for start_idx in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start_idx:start_idx + batch_size]
            queries = [p[0] for p in batch_pairs]
            passages = [p[1] for p in batch_pairs]

            encoded = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)

            logits = outputs.logits

            # Most rerankers output one score. Some output two class logits.
            if logits.shape[-1] == 1:
                scores = logits.squeeze(-1)
            else:
                scores = logits[:, -1]

            all_scores.extend(scores.detach().cpu().float().numpy().tolist())

        return np.asarray(all_scores, dtype=np.float32)


def evaluate_reranker(reranker_config, chunks, candidate_rows):
    method_name = reranker_config["method_name"]
    model_name = reranker_config["model_name"]

    detail_path = RESULTS_DIR / f"detailed_query_results_{method_name}.csv"

    if detail_path.exists():
        log(f"Found existing detailed results for {method_name}. Loading instead of rerunning.")
        df = pd.read_csv(detail_path, encoding="utf-8-sig")
        return summarize_detailed_results(method_name, df), df

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = BATCH_SIZE_GPU if device == "cuda" else BATCH_SIZE_CPU

    reranker = PairReranker(model_name=model_name, device=device)

    log(f"Evaluating {method_name} with batch_size={batch_size}")
    rows = []

    iterator = candidate_rows
    if tqdm is not None:
        iterator = tqdm(candidate_rows, total=len(candidate_rows), desc=method_name)

    for item in iterator:
        question = item["question"]
        relevant_id = item["relevant_chunk_id"]
        candidate_indices = item["candidate_indices"]
        candidates = [chunks[int(i)] for i in candidate_indices]

        pairs = [[question, c.get("text", "")] for c in candidates]

        start = time.perf_counter()
        scores = reranker.predict(pairs, batch_size=batch_size)
        rerank_latency = time.perf_counter() - start

        sorted_local_indices = np.argsort(scores)[::-1]
        ranked_candidates = [candidates[int(i)] for i in sorted_local_indices]
        ranked_ids = [c["chunk_id"] for c in ranked_candidates]
        ranked_scores = [float(scores[int(i)]) for i in sorted_local_indices]

        row = {
            "method": method_name,
            "query_index": item["query_index"],
            "question": question,
            "answer": item.get("answer", ""),
            "relevant_chunk_id": relevant_id,
            "retrieved_chunk_ids": json.dumps(ranked_ids, ensure_ascii=False),
            "reranker_scores": json.dumps(ranked_scores, ensure_ascii=False),
            "bm25_candidate_latency_ms": item["bm25_candidate_latency_ms"],
            "rerank_latency_ms": rerank_latency * 1000,
            "total_latency_ms": item["bm25_candidate_latency_ms"] + (rerank_latency * 1000),
        }

        row = add_metrics_to_row(row, ranked_ids, relevant_id)
        rows.append(row)

        # Safety checkpoint every 100 queries.
        if len(rows) % 100 == 0:
            pd.DataFrame(rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    # Free memory before loading next reranker.
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summarize_detailed_results(method_name, df), df


# ============================================================
# Graphs
# ============================================================

def save_bar_chart(df, x_col, y_col, title, ylabel, filename):
    plt.figure(figsize=(11, 6))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def save_grouped_metric_chart(summary_df, metric_prefix, title, filename):
    k_cols = [f"{metric_prefix}@{k}" for k in EVAL_K_VALUES if f"{metric_prefix}@{k}" in summary_df.columns]
    methods = summary_df["method"].tolist()

    x = np.arange(len(methods))
    width = 0.8 / max(len(k_cols), 1)

    plt.figure(figsize=(13, 6))
    for i, col in enumerate(k_cols):
        plt.bar(x + i * width, summary_df[col], width, label=col)

    plt.title(title)
    plt.xlabel("Method")
    plt.ylabel(metric_prefix.upper())
    plt.xticks(x + width * (len(k_cols) - 1) / 2, methods, rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def create_graphs(summary_df):
    log("Generating graphs...")

    save_grouped_metric_chart(
        summary_df,
        "recall",
        "Experiment 2: Recall@K Comparison",
        "experiment_2_recall_at_k_comparison.png"
    )

    save_grouped_metric_chart(
        summary_df,
        "mrr",
        "Experiment 2: MRR@K Comparison",
        "experiment_2_mrr_at_k_comparison.png"
    )

    save_grouped_metric_chart(
        summary_df,
        "ndcg",
        "Experiment 2: nDCG@K Comparison",
        "experiment_2_ndcg_at_k_comparison.png"
    )

    save_bar_chart(
        summary_df.sort_values("recall@1", ascending=False),
        "method",
        "recall@1",
        "Experiment 2: Recall@1 by Method",
        "Recall@1",
        "experiment_2_recall_at_1_by_method.png"
    )

    save_bar_chart(
        summary_df.sort_values("mrr@10", ascending=False),
        "method",
        "mrr@10",
        "Experiment 2: MRR@10 by Method",
        "MRR@10",
        "experiment_2_mrr_at_10_by_method.png"
    )

    save_bar_chart(
        summary_df.sort_values("ndcg@10", ascending=False),
        "method",
        "ndcg@10",
        "Experiment 2: nDCG@10 by Method",
        "nDCG@10",
        "experiment_2_ndcg_at_10_by_method.png"
    )

    save_bar_chart(
        summary_df.sort_values("avg_total_latency_ms"),
        "method",
        "avg_total_latency_ms",
        "Experiment 2: Average Total Latency by Method",
        "Average Total Latency (ms)",
        "experiment_2_avg_total_latency_by_method.png"
    )

    save_bar_chart(
        summary_df.sort_values("avg_rerank_latency_ms"),
        "method",
        "avg_rerank_latency_ms",
        "Experiment 2: Average Reranking Latency by Method",
        "Average Reranking Latency (ms)",
        "experiment_2_avg_rerank_latency_by_method.png"
    )

    save_bar_chart(
        summary_df.sort_values("recall@50", ascending=False),
        "method",
        "recall@50",
        "Experiment 2: Candidate Recall@50 by Method",
        "Recall@50",
        "experiment_2_recall_at_50_by_method.png"
    )


# ============================================================
# Optional comparison with Experiment 1
# ============================================================

def create_combined_comparison_with_experiment_1(exp2_summary_df):
    if not EXPERIMENT1_SUMMARY_PATH.exists():
        log("Experiment-1 summary not found. Skipping combined comparison.")
        return

    try:
        exp1 = pd.read_csv(EXPERIMENT1_SUMMARY_PATH, encoding="utf-8-sig")
    except Exception as e:
        log(f"Could not read Experiment-1 summary: {e}")
        return

    bm25_exp1 = exp1[exp1["method"] == "BM25"].copy()
    if bm25_exp1.empty:
        log("BM25 row not found in Experiment-1 summary. Skipping combined comparison.")
        return

    # Convert Experiment-1 columns to Experiment-2 style.
    row = {
        "method": "Experiment1_BM25",
        "num_queries": int(bm25_exp1.iloc[0].get("num_queries", len(exp2_summary_df))),
        "avg_bm25_candidate_latency_ms": float(bm25_exp1.iloc[0].get("avg_latency_ms", 0.0)),
        "avg_rerank_latency_ms": 0.0,
        "avg_total_latency_ms": float(bm25_exp1.iloc[0].get("avg_latency_ms", 0.0)),
        "median_total_latency_ms": float(bm25_exp1.iloc[0].get("median_latency_ms", 0.0)),
        "max_total_latency_ms": float(bm25_exp1.iloc[0].get("max_latency_ms", 0.0)),
    }

    for k in EVAL_K_VALUES:
        if k == 50:
            # Experiment 1 did not evaluate @50.
            row[f"hit@{k}"] = np.nan
            row[f"recall@{k}"] = np.nan
            row[f"mrr@{k}"] = np.nan
            row[f"ndcg@{k}"] = np.nan
        else:
            row[f"hit@{k}"] = float(bm25_exp1.iloc[0].get(f"hit@{k}", np.nan))
            row[f"recall@{k}"] = float(bm25_exp1.iloc[0].get(f"recall@{k}", np.nan))
            row[f"mrr@{k}"] = float(bm25_exp1.iloc[0].get(f"mrr@{k}", np.nan))
            row[f"ndcg@{k}"] = float(bm25_exp1.iloc[0].get(f"ndcg@{k}", np.nan))

    combined = pd.concat([pd.DataFrame([row]), exp2_summary_df], ignore_index=True)
    combined.to_csv(
        RESULTS_DIR / "combined_experiment_1_and_2_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Combined graphs excluding @50 since Experiment-1 does not have it.
    old_eval_values = EVAL_K_VALUES.copy()
    try:
        globals()["EVAL_K_VALUES"] = [1, 3, 5, 10]
        save_grouped_metric_chart(
            combined,
            "recall",
            "Experiment 1 BM25 vs Experiment 2 Reranking: Recall@K",
            "combined_exp1_exp2_recall_at_k.png"
        )
        save_grouped_metric_chart(
            combined,
            "mrr",
            "Experiment 1 BM25 vs Experiment 2 Reranking: MRR@K",
            "combined_exp1_exp2_mrr_at_k.png"
        )
        save_grouped_metric_chart(
            combined,
            "ndcg",
            "Experiment 1 BM25 vs Experiment 2 Reranking: nDCG@K",
            "combined_exp1_exp2_ndcg_at_k.png"
        )
    finally:
        globals()["EVAL_K_VALUES"] = old_eval_values

    log("Combined Experiment-1 and Experiment-2 comparison saved.")


# ============================================================
# Main
# ============================================================

def main():
    total_start = time.perf_counter()

    # Clear previous log for this run.
    log("============================================================")
    log("Starting Experiment 2: BM25 Top-50 Retrieval + Reranking")
    log("============================================================")
    log(f"Python version: {platform.python_version()}")
    log(f"Platform: {platform.platform()}")
    log(f"Torch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")

    chunks_path = resolve_path(CHUNKS_PATH, FALLBACK_CHUNKS_PATH)
    qa_path = resolve_path(QA_PATH, FALLBACK_QA_PATH)

    log(f"Chunks path: {chunks_path.resolve()}")
    log(f"QA path: {qa_path.resolve()}")

    log("Loading chunks...")
    chunks = load_jsonl(chunks_path)

    log("Loading QA pairs...")
    qa_pairs = load_json(qa_path)

    if MAX_QUERIES is not None:
        log(f"Debug mode active: using only first {MAX_QUERIES} QA pairs.")
        qa_pairs = qa_pairs[:MAX_QUERIES]

    log(f"Total chunks: {len(chunks):,}")
    log(f"Total QA pairs: {len(qa_pairs):,}")

    # Validate gold chunks.
    chunk_id_set = set(c["chunk_id"] for c in chunks)
    missing_gold = [qa["chunk_id"] for qa in qa_pairs if qa["chunk_id"] not in chunk_id_set]
    if missing_gold:
        log(f"WARNING: {len(missing_gold)} QA gold chunk IDs are missing from the corpus.")
    else:
        log("All QA gold chunk IDs exist in the corpus.")

    stats = save_dataset_statistics(chunks, qa_pairs)
    log(f"Dataset statistics saved: {stats}")

    bm25, bm25_build_time, bm25_status = load_or_build_bm25(chunks)
    log(f"BM25 status: {bm25_status}")
    log(f"BM25 build time in this experiment: {bm25_build_time:.2f} seconds")

    candidate_rows = generate_or_load_bm25_candidates(bm25, chunks, qa_pairs)

    all_summaries = []

    bm25_summary, bm25_detail_df = evaluate_bm25_top50_candidate_order(candidate_rows)
    bm25_summary["bm25_build_time_sec"] = bm25_build_time
    bm25_summary["reranker_model_name"] = "none"
    bm25_summary["description"] = "BM25 top-50 candidate order without reranking."
    all_summaries.append(bm25_summary)

    # Save BM25 top50 summary immediately.
    pd.DataFrame(all_summaries).to_csv(
        RESULTS_DIR / "retrieval_summary_experiment_2_partial.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for reranker_config in RERANKERS:
        try:
            summary, detail_df = evaluate_reranker(reranker_config, chunks, candidate_rows)
            summary["bm25_build_time_sec"] = bm25_build_time
            summary["reranker_model_name"] = reranker_config["model_name"]
            summary["description"] = reranker_config["description"]
            all_summaries.append(summary)

            # Save progress after each reranker.
            pd.DataFrame(all_summaries).to_csv(
                RESULTS_DIR / "retrieval_summary_experiment_2_partial.csv",
                index=False,
                encoding="utf-8-sig"
            )

        except Exception as e:
            log(f"ERROR while running {reranker_config['method_name']}: {repr(e)}")
            log("The script will continue with any completed results.")
            error_row = {
                "method": reranker_config["method_name"],
                "num_queries": len(qa_pairs),
                "error": repr(e),
                "reranker_model_name": reranker_config["model_name"],
                "description": reranker_config["description"],
            }
            pd.DataFrame([error_row]).to_csv(
                RESULTS_DIR / f"ERROR_{reranker_config['method_name']}.csv",
                index=False,
                encoding="utf-8-sig"
            )

    summary_df = pd.DataFrame(all_summaries)

    # Reorder important columns first.
    important_cols = [
        "method", "num_queries",
        "recall@1", "recall@3", "recall@5", "recall@10", "recall@50",
        "mrr@1", "mrr@3", "mrr@5", "mrr@10", "mrr@50",
        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@50",
        "avg_bm25_candidate_latency_ms", "avg_rerank_latency_ms",
        "avg_total_latency_ms", "median_total_latency_ms", "max_total_latency_ms",
        "bm25_build_time_sec", "reranker_model_name", "description"
    ]

    existing_cols = [c for c in important_cols if c in summary_df.columns]
    other_cols = [c for c in summary_df.columns if c not in existing_cols]
    summary_df = summary_df[existing_cols + other_cols]

    summary_path = RESULTS_DIR / "retrieval_summary_experiment_2.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    log(f"Final Experiment-2 summary saved to {summary_path}")

    create_graphs(summary_df)
    create_combined_comparison_with_experiment_1(summary_df)

    total_time = time.perf_counter() - total_start
    log(f"Experiment 2 finished in {total_time:.2f} seconds.")

    print("\n================ EXPERIMENT 2 SUMMARY ================")
    print(summary_df)
    print("======================================================\n")
    print(f"Results saved in: {RESULTS_DIR.resolve()}")
    print(f"Figures saved in: {FIGURES_DIR.resolve()}")
    print(f"Logs saved in: {LOG_DIR.resolve()}")
    print(f"Candidate cache saved in: {CACHE_DIR.resolve()}")


if __name__ == "__main__":
    main()
