import json
import re

fixed_data = []

with open("qa_pairs.jsonl", "r", encoding="utf-8") as f:
    raw = f.read()

# Fix common issues:
raw = raw.replace(",\n}", "\n}")   # remove trailing comma before }
raw = raw.replace(", }", " }")
raw = raw.replace(",]", "]")

# Now try parsing
try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    print("Still broken JSON:", e)
    exit()

# Save clean version
with open("qa_pairs_clean.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Cleaned file saved as qa_pairs_clean.json")
print("Total QA pairs:", len(data))