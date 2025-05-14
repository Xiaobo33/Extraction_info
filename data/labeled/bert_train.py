# monkey patch to remove bad argument
import builtins
real_accelerator_init = None

def patch_accelerator():
    from accelerate import Accelerator
    import inspect

    global real_accelerator_init
    real_accelerator_init = Accelerator.__init__

    def safe_init(self, *args, **kwargs):
        if 'use_seedable_sampler' in kwargs:
            del kwargs['use_seedable_sampler']
        return real_accelerator_init(self, *args, **kwargs)

    Accelerator.__init__ = safe_init

patch_accelerator()

# normal imports
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
import argparse
import os
import transformers
print("Transformers version:", transformers.__version__)
print("Loaded from:", transformers.__file__)




def load_csv_as_dataset(train_path, dev_path, test_path, avec_neutre=True):
    """
    Fonction pour charger les fichiers CSV et les convertir en Dataset HuggingFace.
    """
    # Lire les fichiers csv
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    # Supprimer les avis neutres si on ne les garde pas
    if not avec_neutre:
        train_df = train_df[train_df['label'] != 'neutre']
        dev_df = dev_df[dev_df['label'] != 'neutre']
        test_df = test_df[test_df['label'] != 'neutre']

    # Encoder les étiquettes en entiers
    label_encoder = LabelEncoder()
    train_df['labels'] = label_encoder.fit_transform(train_df['label'])
    dev_df['labels'] = label_encoder.transform(dev_df['label'])
    test_df['labels'] = label_encoder.transform(test_df['label'])

    label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # convertir les données en Dataset HuggingFace
    train_dataset = Dataset.from_pandas(train_df[['clean_text', 'labels']])
    dev_dataset = Dataset.from_pandas(dev_df[['clean_text', 'labels']])
    test_dataset = Dataset.from_pandas(test_df[['clean_text', 'labels']])

    return train_dataset, dev_dataset, test_dataset, label_map, label_encoder, test_df

def compute_metrics(eval_pred):
    """
    Fonction pour calculer les métriques pendant l'entraînement du modèle.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }


def main(train_file, dev_file, test_file, avec_neutre):

    # On a choisi une version légère du modèle BERT pré-trainé
    model_name = "distilbert-base-uncased"

    # charger les données
    train_dataset, dev_dataset, test_dataset, label_map, label_encoder, test_df_raw = load_csv_as_dataset(
        train_file, dev_file, test_file, avec_neutre
    )

    num_labels = len(label_map)
    # charger le tokenizer BERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenizer chaque ligne
    def tokenize(example):
    # éviterles valeurs null 
        texts = example["clean_text"]
        if isinstance(texts, list):
            texts = [str(t) if t is not None else "" for t in texts]
        else:
            texts = str(texts) if texts is not None else ""
        return tokenizer(texts, truncation=True, padding="max_length", max_length=128)
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset = dev_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)


    # Convertir en tenseurs PyTorch et sélectionner les colonnes utiles
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Charger le modèle BERT pour classification
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Les arguments d'entraînement
    training_args = TrainingArguments(
        output_dir="./bert_output",
        evaluation_strategy="epoch",
        logging_strategy="epoch",   # Enregistrer les logs à chaque époque
        save_strategy="no",         # Ne sauvegarde pas les checkpoints
        num_train_epochs=2,
        max_steps=800,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none"            # N'envoie pas de logs à tensorboard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()


    # === Evaluation sur DEV ===
    print("[DEV] Résultats de validation :")
    dev_predictions = trainer.predict(dev_dataset)
    y_dev_pred = np.argmax(dev_predictions.predictions, axis=1)
    y_dev_true = dev_dataset["labels"]
    dev_report = classification_report(y_dev_true, y_dev_pred, target_names=label_encoder.classes_)
    dev_cm = confusion_matrix(y_dev_true, y_dev_pred)
    print(dev_report)
    print(dev_cm)

    # === Evaluation sur TEST ===
    print("[TEST] Résultats de test :")
    test_predictions = trainer.predict(test_dataset)
    y_test_pred = np.argmax(test_predictions.predictions, axis=1)
    y_test_true = test_dataset["labels"]
    test_report = classification_report(y_test_true, y_test_pred, target_names=label_encoder.classes_)
    test_cm = confusion_matrix(y_test_true, y_test_pred)
    print(test_report)
    print(test_cm)

    # Nom du fichier de sortie
    if "Books" in train_file:           
        source = "Books"
    elif "Kindle" in train_file:
        source = "Kindle"
    neutre_flag = "avec_neutre" if avec_neutre else "sans_neutre"     # Ajouter suffixe
    result_file = f"{source}_bert_{neutre_flag}.txt"


    # Sauvegarder les résultats dans un fichier texte
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("[DEV] Résultats de validation :\n")
        f.write(dev_report)
        f.write("\nMatrice de confusion (DEV):\n")
        f.write(str(dev_cm))
        f.write("\n\n")

        f.write("[TEST] Résultats de test :\n")
        f.write(test_report)
        f.write("\nMatrice de confusion (TEST):\n")
        f.write(str(test_cm))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Chemin vers le fichier d'entraînement")
    parser.add_argument("--dev", type=str, required=True, help="Chemin vers le fichier de validation")
    parser.add_argument("--test", type=str, required=True, help="Chemin vers le fichier de test")
    parser.add_argument("--avec-neutre", action="store_true", help="Inclure ou non la classe neutre")
    args = parser.parse_args()

    main(args.train, args.dev, args.test, args.avec_neutre)


# exemple : python bert_train.py --train Books_2_train.csv  --dev Books_2_dev.csv --test Books_2_test.csv 
