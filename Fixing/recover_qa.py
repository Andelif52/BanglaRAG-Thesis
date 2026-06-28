import json
import re

with open("qa_pairs.jsonl", "r", encoding="utf-8") as f:
    raw = f.read()

# Extract all JSON-like objects inside the file
objects = re.findall(r'\{.*?\}', raw, re.DOTALL)

clean_data = []

for obj in objects:
    try:
        # fix common trailing comma issues
        obj = re.sub(r',\s*}', '}', obj)
        obj = re.sub(r',\s*]', ']', obj)

        clean_data.append(json.loads(obj))
    except:
        continue

# Save cleaned dataset
with open("qa_pairs_clean.json", "w", encoding="utf-8") as f:
    json.dump(clean_data, f, ensure_ascii=False, indent=2)

print("✅ Recovery complete")
print("Total recovered QA pairs:", len(clean_data))