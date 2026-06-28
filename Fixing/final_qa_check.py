import json

print("Loading QA pairs...")

with open("qa_pairs_clean.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

print("Loading corpus chunk IDs...")

corpus_chunk_ids = set()

with open("bnwiki_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        corpus_chunk_ids.add(item["chunk_id"])

# -------------------------
# Tracking
# -------------------------

missing_chunks = set()
invalid_items = []
empty_items = []
duplicate_questions = []

seen_questions = set()

# -------------------------
# Check loop
# -------------------------

for i, item in enumerate(qa_pairs):

    # 1. Check structure validity
    required_fields = ["question", "answer", "chunk_id", "doc_id", "source"]

    if not all(field in item for field in required_fields):
        invalid_items.append((i, item))
        continue

    # 2. Empty fields
    if not item["question"] or not item["answer"]:
        empty_items.append((i, item))

    # 3. Chunk validity
    if item["chunk_id"] not in corpus_chunk_ids:
        missing_chunks.add(item["chunk_id"])

    # 4. Duplicate detection
    q = item["question"].strip()

    if q in seen_questions:
        duplicate_questions.append((i, item))
    else:
        seen_questions.add(q)

# -------------------------
# REPORT
# -------------------------

print("\n================ QA DATASET REPORT ================\n")

print(f"Total QA pairs        : {len(qa_pairs)}")
print(f"Invalid items         : {len(invalid_items)}")
print(f"Empty items           : {len(empty_items)}")
print(f"Missing chunk IDs     : {len(missing_chunks)}")
print(f"Duplicate questions   : {len(duplicate_questions)}")

print("\n===================================================\n")

# -------------------------
# SHOW INVALID ITEMS
# -------------------------

if invalid_items:
    print("\n❌ INVALID ITEMS:\n")
    for idx, item in invalid_items[:10]:
        print(f"Index: {idx}")
        print(json.dumps(item, ensure_ascii=False, indent=2))
        print("-" * 50)

# -------------------------
# SHOW DUPLICATES
# -------------------------

if duplicate_questions:
    print("\n⚠️ DUPLICATE QUESTIONS:\n")
    for idx, item in duplicate_questions[:10]:
        print(f"Index: {idx}")
        print(item["question"])
        print(json.dumps(item, ensure_ascii=False, indent=2))
        print("-" * 50)

# -------------------------
# SUMMARY
# -------------------------

print("\n===================================================\n")

if len(invalid_items) == 0 and len(missing_chunks) == 0:
    print("🎯 Dataset is structurally clean (safe for BM25)")
else:
    print("⚠️ Fix the above issues before running retrieval experiments")