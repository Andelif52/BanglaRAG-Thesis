import json
import re
from tqdm import tqdm

INPUT_FILE = r"F:\Andelif\bnwiki_clean.jsonl"
OUTPUT_FILE = r"F:\Andelif\bnwiki_cleaned.jsonl"


def clean_text(text):

    # remove citation/template blocks
    text = re.sub(r'\{\{.*?\}\}', ' ', text)

    # remove image/file remnants
    text = re.sub(r'\d+px\|', ' ', text)
    text = re.sub(r'alt=', ' ', text)
    text = re.sub(r'link=', ' ', text)

    # remove URLs
    text = re.sub(r'http\S+', ' ', text)

    # remove escaped newlines
    text = re.sub(r'\\n+', ' ', text)

    # remove actual newlines
    text = re.sub(r'\n+', ' ', text)

    # remove excessive spaces/tabs
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in tqdm(infile):

        try:
            article = json.loads(line)

            title = article.get("title", "").strip()
            text = article.get("text", "").strip()

            text = clean_text(text)

            # skip very short/noisy articles
            if len(text) < 300:
                continue

            cleaned_article = {
                "id": f"wiki_{article.get('id')}",
                "title": title,
                "text": text,
                "source": "wikipedia"
            }

            outfile.write(
                json.dumps(cleaned_article, ensure_ascii=False) + "\n"
            )

        except:
            continue

print("Cleaning completed.")