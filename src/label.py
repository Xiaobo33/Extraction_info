import os
import json

def label_jsonl(input_file, output_file):
    total, kept = 0, 0
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            total += 1
            rating = data.get("rating", None)
            
            if rating == 3.0:
                #continue
                data["label"] = "neutre"
            elif rating in [4.0, 5.0]:
                data["label"] = "positive"
            elif rating in [1.0, 2.0]:
                data["label"] = "negative"
            else:
                continue

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept += 1

label_jsonl("Books_cleaned.jsonl", "Books_3_labeled.jsonl")