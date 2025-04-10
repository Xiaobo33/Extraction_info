import json
import pandas as pd

def extract_fields_from_jsonl(file_path):
    """Extraire rating, clean_text, label et enregistrer dans DataFrame"""
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line.strip())

            data.append({
                "rating": review.get("rating"),
                "clean_text": review.get("clean_text", ""),
                "label": review.get("label", "")
            })

    return pd.DataFrame(data)

df = extract_fields_from_jsonl("Kindle_2_labeled.jsonl")

print(df.head())
df.to_csv("Kindle_2_extracted.csv", index=False, encoding="utf-8")


