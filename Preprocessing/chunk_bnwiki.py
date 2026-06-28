import json
from tqdm import tqdm

INPUT_FILE = r"F:\Andelif\bnwiki_cleaned.jsonl"
OUTPUT_FILE = r"F:\Andelif\bnwiki_chunks.jsonl"

CHUNK_SIZE = 256
OVERLAP = 40


def chunk_text(text, chunk_size=256, overlap=40):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in tqdm(infile):

        try:
            article = json.loads(line)

            doc_id = article["id"]
            title = article["title"]
            source = article["source"]
            text = article["text"]

            chunks = chunk_text(
                text,
                CHUNK_SIZE,
                OVERLAP
            )

            for i, chunk in enumerate(chunks):

                if len(chunk.strip()) < 100:
                    continue

                chunk_data = {
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "doc_id": doc_id,
                    "title": title,
                    "text": chunk,
                    "source": source
                }

                outfile.write(
                    json.dumps(chunk_data, ensure_ascii=False) + "\n"
                )

        except:
            continue

print("Chunking completed.")