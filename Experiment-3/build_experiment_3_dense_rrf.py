import os
import json
import time
import argparse
import pickle
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import faiss
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

CHUNKS_PATH = r"C:\Thesis\Andelif\bnwiki_chunks.jsonl"
QA_PATH = r"C:\Thesis\Andelif\qa_pairs_clean.json"

# This is the preferred BM25 source because Experiment 2 already generated it.
BM25_TOP50_CACHE_PATH = r"C:\Thesis\Andelif\Experiment-2\outputs\cache\bm25_top50_candidates.jsonl"

# Fallback if JSONL cache cannot be found.
BM25_DETAILED_CSV_PATH = r"C:\Thesis\Andelif\Experiment-2\outputs\results\detailed_query_results_BM25_Top50_candidate_order.csv"

# Optional fallback if both previous files are missing.
BM25_PICKLE_PATHS = [
    r"C:\Thesis\Andelif\Experiment-2\outputs\indexes\bm25.pkl",
    r"C:\Thesis\Andelif\Experiment-1\outputs\indexes\bm25.pkl",
]

OUTPUT_DIR = "outputs"
EMBEDDING_DIR = os.path.join(OUTPUT_DIR, "embeddings")
INDEX_DIR = os.path.join(OUTPUT_DIR, "indexes")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for d in [OUTPUT_DIR, EMBEDDING_DIR, INDEX_DIR, RESULTS_DIR, GRAPH_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# Experiment settings
# ============================================================

TOP_K_VALUES = [1, 3, 5, 10, 20, 50]
DENSE_RETRIEVAL_TOP_K = 50
BM25_RETRIEVAL_TOP_K = 50
RRF_K = 60

BATCH_SIZE = 16

MODEL_CONFIGS = {
    "e5": {
        "method_dense": "Dense_multilingual_E5_base",
        "method_hybrid": "Hybrid_RRF_BM25_E5_base",
        "model_name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_seq_length": 512,
    },
    "bge": {
        "method_dense": "Dense_BGE_M3",
        "method_hybrid": "Hybrid_RRF_BM25_BGE_M3",
        "model_name": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "max_seq_length": 512,
    },
}


# ============================================================
# Logging
# ============================================================

LOG_FILE = None


def setup_logger(phase):
    global LOG_FILE
    LOG_FILE = os.path.join(LOG_DIR, f"experiment_3_{phase}_log.txt")


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ============================================================
# Loading data
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


# ============================================================
# BM25 candidate loading
# ============================================================

def extract_candidate_ids_from_json_obj(obj):
    possible_keys = [
        "candidate_chunk_ids",
        "retrieved_chunk_ids",
        "top_chunk_ids",
        "chunk_ids",
        "candidates",
        "top_candidates",
        "results",
        "retrieved",
    ]

    for key in possible_keys:
        if key not in obj:
            continue

        value = obj[key]

        if isinstance(value, list):
            ids = []

            for x in value:
                if isinstance(x, str):
                    ids.append(x)
                elif isinstance(x, int):
                    ids.append(str(x))
                elif isinstance(x, dict):
                    cid = (
                        x.get("chunk_id")
                        or x.get("id")
                        or x.get("candidate_chunk_id")
                        or x.get("doc_id")
                    )
                    if cid is not None:
                        ids.append(str(cid))

            if ids:
                return ids

    return []


def load_bm25_candidates_from_jsonl(path, qa_pairs):
    log(f"Trying to load BM25 top candidates from JSONL cache: {path}")

    if not os.path.exists(path):
        log("BM25 JSONL cache not found.")
        return None

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            ids = extract_candidate_ids_from_json_obj(obj)

            if ids:
                rows.append(ids[:BM25_RETRIEVAL_TOP_K])

    if len(rows) != len(qa_pairs):
        log(f"BM25 JSONL row count mismatch. Rows: {len(rows)}, QA pairs: {len(qa_pairs)}")
        log("Will try another BM25 source.")
        return None

    log(f"Loaded BM25 candidates from JSONL for {len(rows):,} queries.")
    return rows


def load_bm25_candidates_from_detailed_csv(path, qa_pairs):
    log(f"Trying to load BM25 top candidates from detailed CSV: {path}")

    if not os.path.exists(path):
        log("BM25 detailed CSV not found.")
        return None

    df = pd.read_csv(path)

    rank_cols = []
    for col in df.columns:
        lower = col.lower()
        if lower.startswith("rank_") and lower.endswith("_chunk_id"):
            rank_cols.append(col)

    def rank_number(col):
        parts = col.split("_")
        for p in parts:
            if p.isdigit():
                return int(p)
        return 999999

    rank_cols = sorted(rank_cols, key=rank_number)
    rank_cols = rank_cols[:BM25_RETRIEVAL_TOP_K]

    if not rank_cols:
        log("No rank_N_chunk_id columns found in detailed CSV.")
        return None

    rows = []
    for _, row in df.iterrows():
        ids = []
        for col in rank_cols:
            value = row[col]
            if pd.notna(value):
                ids.append(str(value))
        rows.append(ids)

    if len(rows) != len(qa_pairs):
        log(f"BM25 CSV row count mismatch. Rows: {len(rows)}, QA pairs: {len(qa_pairs)}")
        log("Will try another BM25 source.")
        return None

    log(f"Loaded BM25 candidates from detailed CSV for {len(rows):,} queries.")
    return rows


def simple_bangla_tokenize(text):
    return str(text).split()


def compute_bm25_candidates_from_pickle(chunks, qa_pairs):
    log("Trying to compute BM25 candidates from pickle. This can be slow.")

    bm25_path = None
    for p in BM25_PICKLE_PATHS:
        if os.path.exists(p):
            bm25_path = p
            break

    if bm25_path is None:
        raise FileNotFoundError(
            "Could not find BM25 candidates or BM25 pickle. "
            "Please check BM25_TOP50_CACHE_PATH or BM25_DETAILED_CSV_PATH."
        )

    log(f"Loading BM25 pickle: {bm25_path}")

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    all_rows = []
    start = time.time()

    for qa in tqdm(qa_pairs, desc="Computing BM25 top candidates"):
        query_tokens = simple_bangla_tokenize(qa["question"])
        scores = bm25.get_scores(query_tokens)

        top_idx = np.argpartition(scores, -BM25_RETRIEVAL_TOP_K)[-BM25_RETRIEVAL_TOP_K:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        ids = [chunks[i]["chunk_id"] for i in top_idx]
        all_rows.append(ids)

    elapsed = time.time() - start
    log(f"Computed BM25 candidates in {elapsed:.2f} seconds.")

    return all_rows


def load_or_compute_bm25_candidates(chunks, qa_pairs):
    rows = load_bm25_candidates_from_jsonl(BM25_TOP50_CACHE_PATH, qa_pairs)
    if rows is not None:
        return rows

    rows = load_bm25_candidates_from_detailed_csv(BM25_DETAILED_CSV_PATH, qa_pairs)
    if rows is not None:
        return rows

    return compute_bm25_candidates_from_pickle(chunks, qa_pairs)


# ============================================================
# Embedding and FAISS
# ============================================================

def normalize_embeddings(x):
    x = np.asarray(x).astype("float32")
    faiss.normalize_L2(x)
    return x


def get_embedding_paths(phase):
    emb_path = os.path.join(EMBEDDING_DIR, f"corpus_embeddings_{phase}.npy")
    ids_path = os.path.join(EMBEDDING_DIR, f"corpus_chunk_ids_{phase}.json")
    return emb_path, ids_path


def encode_or_load_corpus_embeddings(model, chunks, config, phase):
    emb_path, ids_path = get_embedding_paths(phase)

    if os.path.exists(emb_path) and os.path.exists(ids_path):
        log(f"Found existing corpus embeddings: {emb_path}")
        embeddings = np.load(emb_path)

        with open(ids_path, "r", encoding="utf-8") as f:
            chunk_ids = json.load(f)

        log(f"Loaded embeddings shape: {embeddings.shape}")
        return embeddings, chunk_ids, 0.0

    log("No existing corpus embeddings found.")
    log("Encoding full corpus. This may take a long time.")

    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [
        config["passage_prefix"] + c["text"].replace("\n", " ").strip()
        for c in chunks
    ]

    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = normalize_embeddings(embeddings)
    encode_time = time.time() - start

    log(f"Corpus encoding finished in {encode_time:.2f} seconds.")
    log(f"Embedding shape: {embeddings.shape}")

    np.save(emb_path, embeddings)

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)

    log(f"Saved corpus embeddings: {emb_path}")
    log(f"Saved corpus chunk ids: {ids_path}")

    return embeddings, chunk_ids, encode_time


def build_or_load_faiss_index(embeddings, phase):
    index_path = os.path.join(INDEX_DIR, f"faiss_index_{phase}.index")

    if os.path.exists(index_path):
        log(f"Found existing FAISS index: {index_path}")
        index = faiss.read_index(index_path)
        return index, 0.0

    log("Building FAISS IndexFlatIP index.")
    start = time.time()

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    build_time = time.time() - start

    faiss.write_index(index, index_path)

    log(f"FAISS index built in {build_time:.2f} seconds.")
    log(f"Saved FAISS index: {index_path}")

    return index, build_time


def encode_queries(model, qa_pairs, config):
    questions = [
        config["query_prefix"] + qa["question"].replace("\n", " ").strip()
        for qa in qa_pairs
    ]

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

    avg_query_encode_latency_ms = (query_encode_time / len(qa_pairs)) * 1000

    log(f"Query encoding time: {query_encode_time:.2f} seconds.")
    log(f"Average query encoding latency: {avg_query_encode_latency_ms:.2f} ms/query.")

    return query_embeddings, query_encode_time, avg_query_encode_latency_ms


def run_dense_search(index, query_embeddings, chunk_ids):
    log(f"Running dense FAISS search top-{DENSE_RETRIEVAL_TOP_K}.")
    start = time.time()

    scores, indices = index.search(query_embeddings, DENSE_RETRIEVAL_TOP_K)

    search_time = time.time() - start
    avg_search_latency_ms = (search_time / query_embeddings.shape[0]) * 1000

    log(f"Dense search time: {search_time:.2f} seconds.")
    log(f"Average dense search latency: {avg_search_latency_ms:.2f} ms/query.")

    dense_rows = []

    for i in range(indices.shape[0]):
        ids = [chunk_ids[idx] for idx in indices[i].tolist()]
        sims = scores[i].tolist()

        dense_rows.append({
            "retrieved_chunk_ids": ids,
            "scores": sims,
        })

    return dense_rows, search_time, avg_search_latency_ms


# ============================================================
# RRF fusion
# ============================================================

def reciprocal_rank_fusion(bm25_ids, dense_ids, rrf_k=60, top_k=50):
    scores = defaultdict(float)

    for rank, cid in enumerate(bm25_ids, start=1):
        scores[cid] += 1.0 / (rrf_k + rank)

    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] += 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_ids = [cid for cid, _ in ranked[:top_k]]
    ranked_scores = [score for _, score in ranked[:top_k]]

    return ranked_ids, ranked_scores


def run_rrf_hybrid(bm25_rows, dense_rows):
    log("Running RRF hybrid fusion.")

    hybrid_rows = []

    start = time.time()

    for bm25_ids, dense_row in tqdm(
        zip(bm25_rows, dense_rows),
        total=len(dense_rows),
        desc="RRF fusion"
    ):
        dense_ids = dense_row["retrieved_chunk_ids"]

        fused_ids, fused_scores = reciprocal_rank_fusion(
            bm25_ids=bm25_ids,
            dense_ids=dense_ids,
            rrf_k=RRF_K,
            top_k=DENSE_RETRIEVAL_TOP_K,
        )

        hybrid_rows.append({
            "retrieved_chunk_ids": fused_ids,
            "scores": fused_scores,
        })

    fusion_time = time.time() - start
    avg_fusion_latency_ms = (fusion_time / len(dense_rows)) * 1000

    log(f"RRF fusion time: {fusion_time:.2f} seconds.")
    log(f"Average RRF fusion latency: {avg_fusion_latency_ms:.4f} ms/query.")

    return hybrid_rows, fusion_time, avg_fusion_latency_ms


# ============================================================
# Evaluation
# ============================================================

def evaluate_rows(method_name, qa_pairs, retrieval_rows):
    metrics = {
        "method": method_name,
        "num_queries": len(qa_pairs),
    }

    for k in TOP_K_VALUES:
        hits = 0
        rr_sum = 0.0
        ndcg_sum = 0.0

        for qa, row in zip(qa_pairs, retrieval_rows):
            gold = qa["gold_chunk_id"]
            retrieved = row["retrieved_chunk_ids"][:k]

            if gold in retrieved:
                hits += 1
                rank = retrieved.index(gold) + 1
                rr_sum += 1.0 / rank
                ndcg_sum += 1.0 / np.log2(rank + 1)
            else:
                rr_sum += 0.0
                ndcg_sum += 0.0

        metrics[f"recall@{k}"] = hits / len(qa_pairs)
        metrics[f"mrr@{k}"] = rr_sum / len(qa_pairs)
        metrics[f"ndcg@{k}"] = ndcg_sum / len(qa_pairs)

    return metrics


def build_detailed_results(method_name, qa_pairs, retrieval_rows):
    detailed = []

    for qa, row in zip(qa_pairs, retrieval_rows):
        retrieved_ids = row["retrieved_chunk_ids"]
        scores = row["scores"]
        gold = qa["gold_chunk_id"]

        gold_found_rank = None
        if gold in retrieved_ids:
            gold_found_rank = retrieved_ids.index(gold) + 1

        out = {
            "method": method_name,
            "qa_index": qa["qa_index"],
            "qa_id": qa["qa_id"],
            "question": qa["question"],
            "answer": qa["answer"],
            "gold_chunk_id": gold,
            "gold_found_rank": gold_found_rank,
        }

        for rank in range(1, DENSE_RETRIEVAL_TOP_K + 1):
            if rank <= len(retrieved_ids):
                out[f"rank_{rank}_chunk_id"] = retrieved_ids[rank - 1]
                out[f"rank_{rank}_score"] = scores[rank - 1]
            else:
                out[f"rank_{rank}_chunk_id"] = ""
                out[f"rank_{rank}_score"] = ""

        detailed.append(out)

    return pd.DataFrame(detailed)


# ============================================================
# Graphs
# ============================================================

def plot_metric_at_k(summary_df, metric_prefix, title, output_path):
    plt.figure(figsize=(10, 6))

    x = TOP_K_VALUES

    for _, row in summary_df.iterrows():
        y = [row.get(f"{metric_prefix}@{k}", np.nan) for k in x]
        plt.plot(x, y, marker="o", label=row["method"])

    plt.xlabel("K")
    plt.ylabel(metric_prefix.upper())
    plt.title(title)
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_graphs(summary_df, phase):
    recall_path = os.path.join(GRAPH_DIR, f"experiment_3_{phase}_recall_at_k.png")
    mrr_path = os.path.join(GRAPH_DIR, f"experiment_3_{phase}_mrr_at_k.png")
    ndcg_path = os.path.join(GRAPH_DIR, f"experiment_3_{phase}_ndcg_at_k.png")

    plot_metric_at_k(
        summary_df,
        "recall",
        f"Experiment 3 {phase.upper()}: Recall@K",
        recall_path,
    )

    plot_metric_at_k(
        summary_df,
        "mrr",
        f"Experiment 3 {phase.upper()}: MRR@K",
        mrr_path,
    )

    plot_metric_at_k(
        summary_df,
        "ndcg",
        f"Experiment 3 {phase.upper()}: nDCG@K",
        ndcg_path,
    )

    log(f"Saved graph: {recall_path}")
    log(f"Saved graph: {mrr_path}")
    log(f"Saved graph: {ndcg_path}")


# ============================================================
# Main phase runner
# ============================================================

def run_phase(phase):
    setup_logger(phase)

    config = MODEL_CONFIGS[phase]

    log("=" * 80)
    log(f"Starting Experiment 3 phase: {phase}")
    log(f"Dense method: {config['method_dense']}")
    log(f"Hybrid method: {config['method_hybrid']}")
    log(f"Model: {config['model_name']}")
    log("=" * 80)

    log(f"Python version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")

    experiment_start = time.time()

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

    bm25_rows = load_or_compute_bm25_candidates(chunks, qa_pairs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading sentence-transformer model on device: {device}")

    model = SentenceTransformer(config["model_name"], device=device)
    model.max_seq_length = config["max_seq_length"]

    corpus_embeddings, corpus_chunk_ids, corpus_encode_time = encode_or_load_corpus_embeddings(
        model=model,
        chunks=chunks,
        config=config,
        phase=phase,
    )

    index, index_build_time = build_or_load_faiss_index(
        embeddings=corpus_embeddings,
        phase=phase,
    )

    query_embeddings, query_encode_time, avg_query_encode_latency_ms = encode_queries(
        model=model,
        qa_pairs=qa_pairs,
        config=config,
    )

    dense_rows, dense_search_time, avg_dense_search_latency_ms = run_dense_search(
        index=index,
        query_embeddings=query_embeddings,
        chunk_ids=corpus_chunk_ids,
    )

    dense_metrics = evaluate_rows(
        method_name=config["method_dense"],
        qa_pairs=qa_pairs,
        retrieval_rows=dense_rows,
    )

    dense_metrics.update({
        "model_name": config["model_name"],
        "phase": phase,
        "corpus_size": len(chunks),
        "top_k": DENSE_RETRIEVAL_TOP_K,
        "corpus_encode_time_sec": corpus_encode_time,
        "index_build_time_sec": index_build_time,
        "query_encode_time_sec": query_encode_time,
        "dense_search_time_sec": dense_search_time,
        "avg_query_encode_latency_ms": avg_query_encode_latency_ms,
        "avg_dense_search_latency_ms": avg_dense_search_latency_ms,
        "avg_total_dense_latency_ms": avg_query_encode_latency_ms + avg_dense_search_latency_ms,
        "rrf_k": "",
    })

    hybrid_rows, fusion_time, avg_fusion_latency_ms = run_rrf_hybrid(
        bm25_rows=bm25_rows,
        dense_rows=dense_rows,
    )

    hybrid_metrics = evaluate_rows(
        method_name=config["method_hybrid"],
        qa_pairs=qa_pairs,
        retrieval_rows=hybrid_rows,
    )

    hybrid_metrics.update({
        "model_name": config["model_name"],
        "phase": phase,
        "corpus_size": len(chunks),
        "top_k": DENSE_RETRIEVAL_TOP_K,
        "corpus_encode_time_sec": corpus_encode_time,
        "index_build_time_sec": index_build_time,
        "query_encode_time_sec": query_encode_time,
        "dense_search_time_sec": dense_search_time,
        "rrf_fusion_time_sec": fusion_time,
        "avg_query_encode_latency_ms": avg_query_encode_latency_ms,
        "avg_dense_search_latency_ms": avg_dense_search_latency_ms,
        "avg_rrf_fusion_latency_ms": avg_fusion_latency_ms,
        "avg_total_dense_plus_rrf_latency_ms": (
            avg_query_encode_latency_ms
            + avg_dense_search_latency_ms
            + avg_fusion_latency_ms
        ),
        "rrf_k": RRF_K,
    })

    summary_df = pd.DataFrame([dense_metrics, hybrid_metrics])

    summary_path = os.path.join(
        RESULTS_DIR,
        f"retrieval_summary_experiment_3_{phase}.csv"
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    log(f"Saved phase summary: {summary_path}")

    dense_detailed_df = build_detailed_results(
        method_name=config["method_dense"],
        qa_pairs=qa_pairs,
        retrieval_rows=dense_rows,
    )

    dense_detailed_path = os.path.join(
        RESULTS_DIR,
        f"detailed_query_results_{config['method_dense']}.csv"
    )
    dense_detailed_df.to_csv(dense_detailed_path, index=False, encoding="utf-8-sig")
    log(f"Saved dense detailed results: {dense_detailed_path}")

    hybrid_detailed_df = build_detailed_results(
        method_name=config["method_hybrid"],
        qa_pairs=qa_pairs,
        retrieval_rows=hybrid_rows,
    )

    hybrid_detailed_path = os.path.join(
        RESULTS_DIR,
        f"detailed_query_results_{config['method_hybrid']}.csv"
    )
    hybrid_detailed_df.to_csv(hybrid_detailed_path, index=False, encoding="utf-8-sig")
    log(f"Saved hybrid detailed results: {hybrid_detailed_path}")

    create_graphs(summary_df, phase)

    total_time = time.time() - experiment_start

    log("=" * 80)
    log(f"Experiment 3 phase {phase} finished.")
    log(f"Total phase time: {total_time:.2f} seconds.")
    log("=" * 80)

    print("\nFinal Summary:")
    print(summary_df.to_string(index=False))


def combine_phase_summaries():
    summary_files = [
        os.path.join(RESULTS_DIR, "retrieval_summary_experiment_3_e5.csv"),
        os.path.join(RESULTS_DIR, "retrieval_summary_experiment_3_bge.csv"),
    ]

    dfs = []

    for path in summary_files:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    if not dfs:
        log("No phase summaries found to combine.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    combined_path = os.path.join(RESULTS_DIR, "retrieval_summary_experiment_3_combined.csv")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")

    log(f"Saved combined Experiment 3 summary: {combined_path}")

    create_graphs(combined, "combined")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["e5", "bge", "both", "combine"],
        required=True,
        help="Which phase to run: e5, bge, both, or combine",
    )

    args = parser.parse_args()

    if args.phase == "e5":
        run_phase("e5")
        combine_phase_summaries()

    elif args.phase == "bge":
        run_phase("bge")
        combine_phase_summaries()

    elif args.phase == "both":
        run_phase("e5")
        run_phase("bge")
        combine_phase_summaries()

    elif args.phase == "combine":
        setup_logger("combine")
        combine_phase_summaries()


if __name__ == "__main__":
    main()