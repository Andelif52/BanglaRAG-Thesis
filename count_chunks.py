import json

chunk_count = 0
doc_ids = set()

with open("bnwiki_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunk_count += 1

            data = json.loads(line)
            doc_ids.add(data["doc_id"])

print("=" * 40)
print(f"Total chunks     : {chunk_count}")
print(f"Total documents  : {len(doc_ids)}")
print(f"Avg chunks/doc   : {chunk_count / len(doc_ids):.2f}")
print("=" * 40)