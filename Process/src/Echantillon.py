import json
import random

input_file = "Books.jsonl"
output_file = "Books_échantillon.jsonl"
sample_size = 1000  # je garde 10000 lignes au hasard dans le corpus

with open(input_file, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

échantillon = random.sample(lines, min(sample_size, len(lines)))

with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.writelines(échantillon)