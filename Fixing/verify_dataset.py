import json

qa_chunk_ids = set()

with open("qa_pairs.jsonl", "r", encoding="utf-8") as f:
    data = json.load(f)   # ✅ correct for JSON array

for item in data:
    qa_chunk_ids.add(item["chunk_id"])

corpus_chunk_ids = set()

with open("bnwiki_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        corpus_chunk_ids.add(item["chunk_id"])

missing = qa_chunk_ids - corpus_chunk_ids

print("=" * 50)
print(f"QA pairs              : {len(qa_chunk_ids)}")
print(f"Corpus chunks         : {len(corpus_chunk_ids)}")
print(f"Missing chunk IDs     : {len(missing)}")
print("=" * 50)

if missing:
    print("\nFirst 10 missing IDs:")
    for x in list(missing)[:10]:
        print(x)
else:
    print("\n✅ Dataset fully consistent!")