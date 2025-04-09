import json
from sklearn.model_selection import train_test_split

data = []
with open("Books_cleaned.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        review = json.loads(line)
        data.append(review)

# on divise les 3 datasets en 80%-10%-10%
train, temp = train_test_split(data, test_size=0.2, random_state=42)
dev, test = train_test_split(temp, test_size=0.5, random_state=42)

# sauvegarde des datasets
def save_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as file:
        for entry in dataset:
            file.write(json.dumps(entry) + "\n")

save_jsonl("Books_train.jsonl", train)
save_jsonl("Books_dev.jsonl", dev)
save_jsonl("Books_test.jsonl", test)

print(f"Mission complete ：train {len(train)}，dev : {len(dev)}，test {len(test)}")