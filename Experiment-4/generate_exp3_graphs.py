import os
import json
import glob
import warnings
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Experiment 3 / 4B Graph Generator
# BM25 candidate expansion + BM25 Top-100 BGE-M3 reranking
# ============================================================

BASE_DIR = Path(r"C:\Thesis\Andelif\Experiment-4")

RESULT_DIRS = [
    BASE_DIR / "outputs" / "results",
    BASE_DIR / "outputs" / "experiment_4B_bm25_top100_bge_m3" / "results",
]

GRAPH_DIR = (
    BASE_DIR
    / "outputs"
    / "experiment_4B_bm25_top100_bge_m3"
    / "graphs"
)

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def normalize_col(col):
    return str(col).strip().lower().replace(" ", "_")


def read_result_file(path):
    path = Path(path)
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")

        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)

        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return pd.DataFrame(data)

            if isinstance(data, dict):
                # common cases: {"results": [...]}, {"summary": [...]}
                for key in ["results", "summary", "final_summary", "metrics"]:
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])

                return pd.json_normalize(data)

        if suffix == ".jsonl":
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return pd.DataFrame(rows)

    except Exception as e:
        warnings.warn(f"Could not read {path}: {e}")

    return None


def load_all_results(result_dirs):
    frames = []

    supported = ["*.csv", "*.tsv", "*.xlsx", "*.xls", "*.json", "*.jsonl"]

    for result_dir in result_dirs:
        if not result_dir.exists():
            print(f"[Warning] Result directory not found: {result_dir}")
            continue

        files = []
        for pattern in supported:
            files.extend(glob.glob(str(result_dir / "**" / pattern), recursive=True))

        for file in files:
            df = read_result_file(file)
            if df is None or df.empty:
                continue

            df.columns = [normalize_col(c) for c in df.columns]
            df["source_file"] = str(file)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No readable result files found. Please check the result directories."
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Convert metric-like columns to numeric when possible
    for col in combined.columns:
        if (
            "recall@" in col
            or "mrr@" in col
            or "ndcg@" in col
            or "latency" in col
            or "time_sec" in col
            or "candidate_top_n" in col
            or "num_queries" in col
        ):
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.drop_duplicates()

    return combined


def get_method_col(df):
    for col in ["method", "model", "retriever", "name"]:
        if col in df.columns:
            return col
    return None


def save_plot(filename):
    output_path = GRAPH_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


def nice_label(row, method_col):
    method = str(row.get(method_col, "Unknown"))

    if "reranker_model" in row and pd.notna(row["reranker_model"]):
        reranker = str(row["reranker_model"])
        if reranker and reranker.lower() != "nan":
            method = f"{method}\n{reranker}"

    if "candidate_top_n" in row and pd.notna(row["candidate_top_n"]):
        candidate_n = int(row["candidate_top_n"])
        if str(candidate_n) not in method:
            method = f"{method}\nTop-{candidate_n}"

    return method


def metric_columns(df, prefix):
    cols = []
    for col in df.columns:
        if col.startswith(prefix + "@"):
            try:
                k = int(col.split("@")[1])
                cols.append((k, col))
            except Exception:
                pass

    cols = sorted(cols, key=lambda x: x[0])
    return cols


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = load_all_results(RESULT_DIRS)

print("\nLoaded result table:")
print(df.head())
print("\nColumns found:")
print(list(df.columns))

method_col = get_method_col(df)

if method_col is None:
    raise ValueError(
        "Could not find a method/model column. Expected one of: method, model, retriever, name"
    )

df[method_col] = df[method_col].astype(str)

# Focus on relevant Experiment-3 / 4B rows
relevant_mask = (
    df[method_col].str.contains("bm25", case=False, na=False)
    | df.astype(str).apply(
        lambda row: row.str.contains("bge", case=False, na=False).any(), axis=1
    )
)

df_rel = df[relevant_mask].copy()

if df_rel.empty:
    df_rel = df.copy()


# ============================================================
# Graph 1: Recall@K curve for available methods
# ============================================================

recall_cols = metric_columns(df_rel, "recall")

if recall_cols:
    plt.figure(figsize=(10, 6))

    # Keep rows that have at least one recall value
    recall_only = df_rel[[method_col] + [c for _, c in recall_cols]].dropna(
        how="all", subset=[c for _, c in recall_cols]
    )

    # Avoid plotting too many duplicate/noisy rows
    recall_only = recall_only.drop_duplicates(subset=[method_col] + [c for _, c in recall_cols])

    for idx, row in recall_only.iterrows():
        x = []
        y = []

        for k, col in recall_cols:
            value = row.get(col)
            if pd.notna(value):
                x.append(k)
                y.append(value)

        if x and y:
            label = nice_label(row, method_col)
            plt.plot(x, y, marker="o", label=label)

    plt.title("Experiment 3: Recall@K Comparison")
    plt.xlabel("K")
    plt.ylabel("Recall")
    plt.xticks([k for k, _ in recall_cols])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    save_plot("exp3_recall_at_k_comparison.png")


# ============================================================
# Graph 2: MRR@K curve
# ============================================================

mrr_cols = metric_columns(df_rel, "mrr")

if mrr_cols:
    plt.figure(figsize=(10, 6))

    mrr_only = df_rel[[method_col] + [c for _, c in mrr_cols]].dropna(
        how="all", subset=[c for _, c in mrr_cols]
    )

    mrr_only = mrr_only.drop_duplicates(subset=[method_col] + [c for _, c in mrr_cols])

    for idx, row in mrr_only.iterrows():
        x = []
        y = []

        for k, col in mrr_cols:
            value = row.get(col)
            if pd.notna(value):
                x.append(k)
                y.append(value)

        if x and y:
            label = nice_label(row, method_col)
            plt.plot(x, y, marker="o", label=label)

    plt.title("Experiment 3: MRR@K Comparison")
    plt.xlabel("K")
    plt.ylabel("MRR")
    plt.xticks([k for k, _ in mrr_cols])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    save_plot("exp3_mrr_at_k_comparison.png")


# ============================================================
# Graph 3: nDCG@K curve
# ============================================================

ndcg_cols = metric_columns(df_rel, "ndcg")

if ndcg_cols:
    plt.figure(figsize=(10, 6))

    ndcg_only = df_rel[[method_col] + [c for _, c in ndcg_cols]].dropna(
        how="all", subset=[c for _, c in ndcg_cols]
    )

    ndcg_only = ndcg_only.drop_duplicates(subset=[method_col] + [c for _, c in ndcg_cols])

    for idx, row in ndcg_only.iterrows():
        x = []
        y = []

        for k, col in ndcg_cols:
            value = row.get(col)
            if pd.notna(value):
                x.append(k)
                y.append(value)

        if x and y:
            label = nice_label(row, method_col)
            plt.plot(x, y, marker="o", label=label)

    plt.title("Experiment 3: nDCG@K Comparison")
    plt.xlabel("K")
    plt.ylabel("nDCG")
    plt.xticks([k for k, _ in ndcg_cols])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    save_plot("exp3_ndcg_at_k_comparison.png")


# ============================================================
# Graph 4: BM25 candidate expansion curve
# Top-50 vs Top-100 vs Top-200 candidate recall
# ============================================================

candidate_points = []

# Case A: rows have candidate_top_n and corresponding recall@N columns
if "candidate_top_n" in df_rel.columns:
    for idx, row in df_rel.iterrows():
        candidate_n = row.get("candidate_top_n")

        if pd.isna(candidate_n):
            continue

        candidate_n = int(candidate_n)
        recall_col = f"recall@{candidate_n}"

        if recall_col in df_rel.columns and pd.notna(row.get(recall_col)):
            method_name = str(row.get(method_col, "")).lower()

            # Candidate expansion should be pure BM25 candidate order, not BGE reranked row
            if "bge" not in method_name and "rerank" not in method_name:
                candidate_points.append(
                    {
                        "candidate_top_n": candidate_n,
                        "recall": row.get(recall_col),
                        "method": row.get(method_col),
                    }
                )

# Case B: one row contains recall@50, recall@100, recall@200
if not candidate_points:
    expansion_cols = []
    for k in [50, 100, 200]:
        col = f"recall@{k}"
        if col in df_rel.columns:
            expansion_cols.append((k, col))

    bm25_rows = df_rel[
        df_rel[method_col].str.contains("bm25", case=False, na=False)
        & ~df_rel[method_col].str.contains("bge", case=False, na=False)
        & ~df_rel[method_col].str.contains("rerank", case=False, na=False)
    ]

    for idx, row in bm25_rows.iterrows():
        for k, col in expansion_cols:
            if pd.notna(row.get(col)):
                candidate_points.append(
                    {
                        "candidate_top_n": k,
                        "recall": row.get(col),
                        "method": row.get(method_col),
                    }
                )

if candidate_points:
    cand_df = pd.DataFrame(candidate_points)
    cand_df = cand_df.drop_duplicates(subset=["candidate_top_n", "recall"])
    cand_df = cand_df.sort_values("candidate_top_n")

    plt.figure(figsize=(8, 5))
    plt.plot(
        cand_df["candidate_top_n"],
        cand_df["recall"],
        marker="o",
    )

    for _, row in cand_df.iterrows():
        plt.text(
            row["candidate_top_n"],
            row["recall"],
            f"{row['recall']:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title("BM25 Candidate Expansion: Recall vs Candidate Pool Size")
    plt.xlabel("BM25 Candidate Pool Size")
    plt.ylabel("Candidate Recall")
    plt.xticks(sorted(cand_df["candidate_top_n"].unique()))
    plt.grid(True, alpha=0.3)
    save_plot("exp3_bm25_candidate_expansion_recall.png")
else:
    print("[Skipped] Candidate expansion graph: no suitable candidate expansion rows found.")


# ============================================================
# Graph 5: Top-100 BM25 Candidate Order vs Top-100 + BGE-M3
# Recall, MRR, nDCG at selected K
# ============================================================

selected_metrics = [
    "recall@1", "recall@3", "recall@5", "recall@10",
    "mrr@10",
    "ndcg@10",
]

available_metrics = [m for m in selected_metrics if m in df_rel.columns]

if available_metrics:
    comparison_rows = df_rel.copy()

    # Prefer Top-100 rows if candidate_top_n exists
    if "candidate_top_n" in comparison_rows.columns:
        top100_rows = comparison_rows[comparison_rows["candidate_top_n"] == 100]
        if not top100_rows.empty:
            comparison_rows = top100_rows

    # Prefer rows that are BM25 candidate order or BGE reranking
    comparison_rows = comparison_rows[
        comparison_rows[method_col].str.contains("bm25", case=False, na=False)
        | comparison_rows[method_col].str.contains("bge", case=False, na=False)
    ]

    comparison_rows = comparison_rows.dropna(how="all", subset=available_metrics)
    comparison_rows = comparison_rows.drop_duplicates(subset=[method_col] + available_metrics)

    if not comparison_rows.empty:
        plot_df = comparison_rows[[method_col] + available_metrics].copy()
        plot_df = plot_df.set_index(method_col)

        ax = plot_df.plot(kind="bar", figsize=(12, 6))
        ax.set_title("Top-100 BM25 Candidate Order vs Top-100 + BGE-M3")
        ax.set_xlabel("Method")
        ax.set_ylabel("Score")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=25, ha="right")
        plt.legend(title="Metric")
        save_plot("exp3_top100_bm25_vs_bge_metrics.png")
    else:
        print("[Skipped] Top-100 comparison graph: no suitable rows found.")


# ============================================================
# Graph 6: Reranking latency / time
# ============================================================

latency_cols = [
    col for col in [
        "avg_rerank_latency_ms",
        "rerank_time_sec",
        "total_experiment_time",
        "total_experiment_time_sec",
    ]
    if col in df_rel.columns
]

if latency_cols:
    latency_rows = df_rel.dropna(how="all", subset=latency_cols).copy()

    if not latency_rows.empty:
        if "candidate_top_n" in latency_rows.columns:
            top100_latency = latency_rows[latency_rows["candidate_top_n"] == 100]
            if not top100_latency.empty:
                latency_rows = top100_latency

        latency_rows = latency_rows.drop_duplicates(subset=[method_col] + latency_cols)

        for latency_col in latency_cols:
            temp = latency_rows[[method_col, latency_col]].dropna()

            if temp.empty:
                continue

            plt.figure(figsize=(9, 5))
            plt.bar(temp[method_col], temp[latency_col])
            plt.title(f"Experiment 3: {latency_col.replace('_', ' ').title()}")
            plt.xlabel("Method")
            plt.ylabel(latency_col.replace("_", " ").title())
            plt.xticks(rotation=25, ha="right")
            plt.grid(True, axis="y", alpha=0.3)

            save_plot(f"exp3_{latency_col}.png")
else:
    print("[Skipped] Latency graph: no latency/time columns found.")


# ============================================================
# Save combined loaded table for checking
# ============================================================

combined_output = GRAPH_DIR / "exp3_loaded_results_combined.csv"
df_rel.to_csv(combined_output, index=False, encoding="utf-8-sig")
print(f"[Saved] {combined_output}")

print("\nDone. Graph generation completed.")
print(f"Graphs saved in: {GRAPH_DIR}")