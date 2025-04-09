import json
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
# j'utilise directement les stopwords de nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)

def clean_text(text):
    # mettre en minuscule
    text = text.lower()
    # supprimer les poncts
    text = re.sub(r'[^\w\s]', '', text)
    # tokenization
    tokens = word_tokenize(text)
    # stopwords
    tokens = [t for t in tokens if t not in stop_words and t not in punct]
    return " ".join(tokens)

def clean_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            raw = data.get("text", "")
            data["clean_text"] = clean_text(raw)
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

clean_jsonl("Kindle_échantillon.jsonl", "Kindle_cleaned.jsonl")