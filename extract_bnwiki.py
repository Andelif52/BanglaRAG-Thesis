import xml.etree.ElementTree as ET
import mwparserfromhell
import json
from tqdm import tqdm

INPUT_XML = r"F:\Andelif\bnwiki-latest-pages-articles.xml"
OUTPUT_FILE = r"F:\Andelif\bnwiki_clean.jsonl"

def clean_wiki_text(text):
    try:
        wikicode = mwparserfromhell.parse(text)
        cleaned = wikicode.strip_code()
        return cleaned.strip()
    except:
        return ""

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:

    context = ET.iterparse(INPUT_XML, events=("end",))

    for event, elem in tqdm(context):

        if elem.tag.endswith("page"):

            title = elem.findtext(".//{*}title")
            ns = elem.findtext(".//{*}ns")
            text = elem.findtext(".//{*}text")
            page_id = elem.findtext(".//{*}id")

            # only main namespace articles
            if ns == "0" and text:

                cleaned_text = clean_wiki_text(text)

                if len(cleaned_text) > 200:

                    article = {
                        "id": page_id,
                        "title": title,
                        "text": cleaned_text
                    }

                    out_file.write(
                        json.dumps(article, ensure_ascii=False) + "\n"
                    )

            elem.clear()

print("Extraction completed.")