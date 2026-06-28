"""
Bangla RAG Retrieval Model Builder + Evaluator
==============================================

This script builds and evaluates:
1. BM25 retrieval
2. Dense retrieval using SentenceTransformer + FAISS Flat index
3. Hybrid retrieval using normalized BM25 + dense scores

Input files expected in the same folder:
- bnwiki_chunks.jsonl
- qa_pairs_clean.json

Outputs:
- outputs/results/retrieval_summary.csv
- outputs/results/detailed_query_results.csv
- outputs/figures/*.png
- outputs/indexes/faiss_flat.index
- outputs/indexes/chunk_metadata.json
- outputs/logs/experiment_log.txt

Run:
    python build_retrieval_models.py
"""

import os
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
from sentence_transformers import SentenceTransformer
import faiss


# =========================
# Configuration
# =========================

CHUNKS_PATH = "bnwiki_chunks.jsonl"
QA_PATH = "qa_pairs_clean.json"

OUTPUT_DIR = Path("outputs")
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
INDEX_DIR = OUTPUT_DIR / "indexes"
LOG_DIR = OUTPUT_DIR / "logs"

for d in [RESULTS_DIR, FIGURES_DIR, INDEX_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TOP_K_VALUES = [1, 3, 5, 10]
DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Hybrid alpha:
# alpha = BM25 weight
# 1-alpha = Dense weight
HYBRID_ALPHAS = [0.25, 0.50, 0.75]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =========================
# Utility Functions
# =========================

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{timestamp}] {message}"
    print(text)
    with open(LOG_DIR / "experiment_log.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def load_jsonl(path):
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bangla_simple_tokenize(text):
    """
    Simple whitespace tokenizer.
    This is intentionally simple for the first BM25 baseline.
    Later we can compare it with improved Bangla tokenization.
    """
    if not isinstance(text, str):
        return []
    return text.split()


def minmax_normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    min_s = float(np.min(scores))
    max_s = float(np.max(scores))
    if max_s - min_s < 1e-9:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - min_s) / (max_s - min_s)


def reciprocal_rank(retrieved_ids, relevant_id):
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid == relevant_id:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids, relevant_id, k):
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if cid == relevant_id:
            return 1.0 / np.log2(i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_id, k):
    # Since each query has one known relevant chunk, ideal DCG is 1.
    return dcg_at_k(retrieved_ids, relevant_id, k)


def evaluate_method(method_name, retrieve_fn, qa_pairs, top_k_values):
    """
    Calculates Hit@K / Recall@K, MRR@K, nDCG@K, latency.
    For this dataset, one question maps to one relevant chunk_id.
    Therefore Hit@K and Recall@K are equivalent.
    """
    log(f"Evaluating {method_name}...")
    max_k = max(top_k_values)

    query_rows = []
    latencies = []

    for idx, qa in enumerate(qa_pairs):
        question = qa["question"]
        relevant_chunk_id = qa["chunk_id"]

        start = time.perf_counter()
        retrieved = retrieve_fn(question, top_k=max_k)
        latency = time.perf_counter() - start
        latencies.append(latency)

        retrieved_ids = [item["chunk_id"] for item in retrieved]

        row = {
            "method": method_name,
            "query_index": idx,
            "question": question,
            "answer": qa.get("answer", ""),
            "relevant_chunk_id": relevant_chunk_id,
            "retrieved_chunk_ids": json.dumps(retrieved_ids, ensure_ascii=False),
            "latency_ms": latency * 1000
        }

        for k in top_k_values:
            top_ids = retrieved_ids[:k]
            row[f"hit@{k}"] = 1 if relevant_chunk_id in top_ids else 0
            row[f"mrr@{k}"] = reciprocal_rank(top_ids, relevant_chunk_id)
            row[f"ndcg@{k}"] = ndcg_at_k(top_ids, relevant_chunk_id, k)

        query_rows.append(row)

    query_df = pd.DataFrame(query_rows)

    summary = {
        "method": method_name,
        "num_queries": len(qa_pairs),
        "avg_latency_ms": float(np.mean(latencies) * 1000),
        "median_latency_ms": float(np.median(latencies) * 1000),
        "min_latency_ms": float(np.min(latencies) * 1000),
        "max_latency_ms": float(np.max(latencies) * 1000),
    }

    for k in top_k_values:
        summary[f"hit@{k}"] = float(query_df[f"hit@{k}"].mean())
        summary[f"recall@{k}"] = float(query_df[f"hit@{k}"].mean())
        summary[f"mrr@{k}"] = float(query_df[f"mrr@{k}"].mean())
        summary[f"ndcg@{k}"] = float(query_df[f"ndcg@{k}"].mean())

    return summary, query_df


# =========================
# Plotting Functions
# =========================

def save_bar_chart(df, x_col, y_col, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def save_grouped_metric_chart(summary_df, metric_prefix, title, filename):
    methods = summary_df["method"].tolist()
    k_cols = [f"{metric_prefix}@{k}" for k in TOP_K_VALUES]

    x = np.arange(len(methods))
    width = 0.8 / len(k_cols)

    plt.figure(figsize=(12, 6))
    for i, col in enumerate(k_cols):
        plt.bar(x + i * width, summary_df[col], width, label=col)

    plt.title(title)
    plt.xlabel("Retrieval Method")
    plt.ylabel(metric_prefix.upper())
    plt.xticks(x + width * (len(k_cols) - 1) / 2, methods, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def save_latency_chart(summary_df):
    sorted_df = summary_df.sort_values("avg_latency_ms")
    save_bar_chart(
        sorted_df,
        x_col="method",
        y_col="avg_latency_ms",
        title="Average Retrieval Latency by Method",
        ylabel="Average Latency (ms)",
        filename="avg_latency_by_method.png"
    )


def save_corpus_stats(chunks, qa_pairs):
    chunk_lengths = [len(c.get("text", "").split()) for c in chunks]
    qa_per_chunk = pd.Series([qa["chunk_id"] for qa in qa_pairs]).value_counts()

    stats = {
        "total_chunks": len(chunks),
        "total_qa_pairs": len(qa_pairs),
        "unique_docs": len(set(c.get("doc_id", "") for c in chunks)),
        "unique_qa_chunks": len(set(qa["chunk_id"] for qa in qa_pairs)),
        "avg_chunk_words": float(np.mean(chunk_lengths)),
        "median_chunk_words": float(np.median(chunk_lengths)),
        "min_chunk_words": int(np.min(chunk_lengths)),
        "max_chunk_words": int(np.max(chunk_lengths)),
        "avg_qa_per_relevant_chunk": float(qa_per_chunk.mean()),
        "max_qa_per_relevant_chunk": int(qa_per_chunk.max())
    }

    pd.DataFrame([stats]).to_csv(RESULTS_DIR / "dataset_statistics.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.hist(chunk_lengths, bins=50)
    plt.title("Chunk Length Distribution")
    plt.xlabel("Chunk Length in Words")
    plt.ylabel("Number of Chunks")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "chunk_length_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(qa_per_chunk.values, bins=30)
    plt.title("QA Pairs per Relevant Chunk Distribution")
    plt.xlabel("Number of QA Pairs per Chunk")
    plt.ylabel("Number of Chunks")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "qa_pairs_per_chunk_distribution.png", dpi=300)
    plt.close()

    return stats


# =========================
# Main Experiment
# =========================

def main():
    start_all = time.perf_counter()

    log("Starting Bangla RAG retrieval experiments.")
    log(f"Python version: {platform.python_version()}")
    log(f"Platform: {platform.platform()}")

    if not Path(CHUNKS_PATH).exists():
        raise FileNotFoundError(f"Could not find {CHUNKS_PATH}. Place it in the same folder as this script.")
    if not Path(QA_PATH).exists():
        raise FileNotFoundError(f"Could not find {QA_PATH}. Place it in the same folder as this script.")

    # Load data
    log("Loading chunks and QA pairs...")
    chunks = load_jsonl(CHUNKS_PATH)
    qa_pairs = load_json(QA_PATH)

    # Basic validation
    required_chunk_keys = {"chunk_id", "doc_id", "title", "text", "source"}
    required_qa_keys = {"question", "answer", "chunk_id", "doc_id", "source"}

    missing_chunk_keys = required_chunk_keys - set(chunks[0].keys())
    missing_qa_keys = required_qa_keys - set(qa_pairs[0].keys())

    if missing_chunk_keys:
        log(f"WARNING: Missing chunk keys: {missing_chunk_keys}")
    if missing_qa_keys:
        log(f"WARNING: Missing QA keys: {missing_qa_keys}")

    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_id_set = set(chunk_ids)
    qa_chunk_id_set = set(qa["chunk_id"] for qa in qa_pairs)
    missing_relevant = qa_chunk_id_set - chunk_id_set

    if missing_relevant:
        log(f"WARNING: {len(missing_relevant)} QA chunk_ids are missing from chunks file.")
    else:
        log("All QA relevant chunk_ids exist in chunk corpus.")

    log(f"Total chunks: {len(chunks)}")
    log(f"Total QA pairs: {len(qa_pairs)}")

    dataset_stats = save_corpus_stats(chunks, qa_pairs)
    log(f"Dataset statistics saved: {dataset_stats}")

    texts = [c.get("text", "") for c in chunks]

    # =========================
    # BM25
    # =========================
    log("Building BM25 index...")
    bm25_start = time.perf_counter()
    tokenized_corpus = [bangla_simple_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_build_time = time.perf_counter() - bm25_start
    log(f"BM25 build time: {bm25_build_time:.2f} seconds")

    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    def retrieve_bm25(query, top_k=10):
        tokenized_query = bangla_simple_tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for i in top_indices:
            item = dict(chunks[int(i)])
            item["score"] = float(scores[int(i)])
            results.append(item)
        return results

    # =========================
    # Dense FAISS Flat
    # =========================
    log(f"Loading dense embedding model: {DENSE_MODEL_NAME}")
    model = SentenceTransformer(DENSE_MODEL_NAME)

    log("Encoding chunk embeddings...")
    dense_start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    embedding_time = time.perf_counter() - dense_start
    log(f"Embedding generation time: {embedding_time:.2f} seconds")

    log("Building FAISS Flat IP index...")
    faiss_start = time.perf_counter()
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    faiss_build_time = time.perf_counter() - faiss_start
    log(f"FAISS Flat build time: {faiss_build_time:.2f} seconds")

    faiss.write_index(faiss_index, str(INDEX_DIR / "faiss_flat.index"))
    np.save(INDEX_DIR / "embeddings.npy", embeddings)

    with open(INDEX_DIR / "chunk_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    def retrieve_dense(query, top_k=10):
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        scores, indices = faiss_index.search(q_emb, top_k)
        results = []
        for score, i in zip(scores[0], indices[0]):
            item = dict(chunks[int(i)])
            item["score"] = float(score)
            results.append(item)
        return results

    # =========================
    # Hybrid
    # =========================
    def make_retrieve_hybrid(alpha):
        def retrieve_hybrid(query, top_k=10):
            # BM25 scores over all chunks
            bm25_scores = bm25.get_scores(bangla_simple_tokenize(query))
            bm25_norm = minmax_normalize(bm25_scores)

            # Dense scores over all chunks using dot product because embeddings are normalized
            q_emb = model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32")
            dense_scores = np.dot(embeddings, q_emb[0])
            dense_norm = minmax_normalize(dense_scores)

            combined = alpha * bm25_norm + (1 - alpha) * dense_norm
            top_indices = np.argsort(combined)[-top_k:][::-1]

            results = []
            for i in top_indices:
                item = dict(chunks[int(i)])
                item["score"] = float(combined[int(i)])
                item["bm25_score_norm"] = float(bm25_norm[int(i)])
                item["dense_score_norm"] = float(dense_norm[int(i)])
                results.append(item)
            return results
        return retrieve_hybrid

    # =========================
    # Evaluate
    # =========================
    all_summaries = []
    all_query_dfs = []

    summary, query_df = evaluate_method("BM25", retrieve_bm25, qa_pairs, TOP_K_VALUES)
    summary["index_build_time_sec"] = bm25_build_time
    summary["embedding_time_sec"] = 0.0
    all_summaries.append(summary)
    all_query_dfs.append(query_df)

    summary, query_df = evaluate_method("Dense_FAISS_Flat", retrieve_dense, qa_pairs, TOP_K_VALUES)
    summary["index_build_time_sec"] = faiss_build_time
    summary["embedding_time_sec"] = embedding_time
    all_summaries.append(summary)
    all_query_dfs.append(query_df)

    for alpha in HYBRID_ALPHAS:
        method_name = f"Hybrid_alpha_{alpha:.2f}"
        retrieve_hybrid = make_retrieve_hybrid(alpha)
        summary, query_df = evaluate_method(method_name, retrieve_hybrid, qa_pairs, TOP_K_VALUES)
        summary["index_build_time_sec"] = bm25_build_time + faiss_build_time
        summary["embedding_time_sec"] = embedding_time
        all_summaries.append(summary)
        all_query_dfs.append(query_df)

    summary_df = pd.DataFrame(all_summaries)
    detailed_df = pd.concat(all_query_dfs, ignore_index=True)

    summary_df.to_csv(RESULTS_DIR / "retrieval_summary.csv", index=False, encoding="utf-8-sig")
    detailed_df.to_csv(RESULTS_DIR / "detailed_query_results.csv", index=False, encoding="utf-8-sig")

    log("Saved result CSV files.")

    # =========================
    # Graphs
    # =========================
    log("Generating graphs...")

    save_grouped_metric_chart(
        summary_df,
        metric_prefix="recall",
        title="Recall@K Comparison Across Retrieval Methods",
        filename="recall_at_k_comparison.png"
    )

    save_grouped_metric_chart(
        summary_df,
        metric_prefix="mrr",
        title="MRR@K Comparison Across Retrieval Methods",
        filename="mrr_at_k_comparison.png"
    )

    save_grouped_metric_chart(
        summary_df,
        metric_prefix="ndcg",
        title="nDCG@K Comparison Across Retrieval Methods",
        filename="ndcg_at_k_comparison.png"
    )

    save_latency_chart(summary_df)

    save_bar_chart(
        summary_df.sort_values("recall@5", ascending=False),
        x_col="method",
        y_col="recall@5",
        title="Recall@5 by Retrieval Method",
        ylabel="Recall@5",
        filename="recall_at_5_by_method.png"
    )

    save_bar_chart(
        summary_df.sort_values("mrr@10", ascending=False),
        x_col="method",
        y_col="mrr@10",
        title="MRR@10 by Retrieval Method",
        ylabel="MRR@10",
        filename="mrr_at_10_by_method.png"
    )

    save_bar_chart(
        summary_df.sort_values("index_build_time_sec"),
        x_col="method",
        y_col="index_build_time_sec",
        title="Index Build Time by Method",
        ylabel="Build Time (seconds)",
        filename="index_build_time_by_method.png"
    )

    total_time = time.perf_counter() - start_all
    log(f"All experiments finished in {total_time:.2f} seconds.")

    print("\n================ RETRIEVAL SUMMARY ================")
    print(summary_df)
    print("===================================================\n")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Indexes saved in: {INDEX_DIR}")


if __name__ == "__main__":
    main()
