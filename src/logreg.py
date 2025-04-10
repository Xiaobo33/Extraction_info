import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import argparse

def load_data(file_path, avec_neutre=True):
    df = pd.read_csv(file_path)
    if not avec_neutre:
        df = df[df['label'] != 'neutre']
    textes = df['clean_text'].fillna("").tolist()
    labels = df['label'].tolist()
    return textes, labels

def entrainer_et_evaluer(train_file, dev_file, test_file, avec_neutre=True):
    print(f"\n=== training（avec_neutre={avec_neutre}）===\n")

    # loadding
    X_train, y_train = load_data(train_file, avec_neutre)
    X_dev, y_dev = load_data(dev_file, avec_neutre)
    X_test, y_test = load_data(test_file, avec_neutre)

    # encoder les labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_dev_encoded = le.transform(y_dev)
    y_test_encoded = le.transform(y_test)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_dev_tfidf = vectorizer.transform(X_dev)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train_encoded)

    # évaluation sur dev
    print("[DEV] Validation resultats:")
    y_dev_pred = model.predict(X_dev_tfidf)
    print(classification_report(y_dev_encoded, y_dev_pred, target_names=le.classes_))
    print(confusion_matrix(y_dev_encoded, y_dev_pred))

    # évaluation sur test
    print("\n[TEST] Test resultats:")
    y_test_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test_encoded, y_test_pred, target_names=le.classes_))
    print(confusion_matrix(y_test_encoded, y_test_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle de classification de sentiments.")
    parser.add_argument("--train", type=str, required=True, help="Chemin vers le fichier CSV d'entraînement")
    parser.add_argument("--dev", type=str, required=True, help="Chemin vers le fichier CSV de validation")
    parser.add_argument("--test", type=str, required=True, help="Chemin vers le fichier CSV de test")
    parser.add_argument("--avec-neutre", action="store_true", help="Inclure les commentaires neutres (label == 'neutre')")

    args = parser.parse_args()

    entrainer_et_evaluer(args.train, args.dev, args.test, avec_neutre=args.avec_neutre)

# exemple : python xx.py --train books_3_train.csv --dev books_3_dev.csv --test books_3_test.csv --avec-neutre