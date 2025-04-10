import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Kindle_2_extracted.csv")

# on divise les 3 datasets en 80%-10%-10%
train, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
dev, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

# sauvegarde des datasets
train.to_csv("Kindle_2_train.csv", index=False)
dev.to_csv("Kindle_2_dev.csv", index=False)
test.to_csv("Kindle_2_test.csv", index=False)

print(f"Mission complete ：train {len(train)}，dev : {len(dev)}，test {len(test)}")