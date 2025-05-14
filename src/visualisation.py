import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# lire les fichiers de résultats
fichiers = {
    "Books (2) - LogReg": "Books_logreg_sans_neutre.txt",
    "Books (2) - BERT": "Books_bert_sans_neutre.txt",
    "Books (3) - LogReg": "Books_logreg_avec_neutre.txt",
    "Books (3) - BERT": "Books_bert_avec_neutre.txt",
    "Kindle (2) - LogReg": "Kindle_logreg_sans_neutre.txt",
    "Kindle (2) - BERT": "Kindle_bert_sans_neutre.txt",
    "Kindle (3) - LogReg": "Kindle_logreg_avec_neutre.txt",
    "Kindle (3) - BERT": "Kindle_bert_avec_neutre.txt",
}

# extraire accuracy et F1 macro
def extraire_scores(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    accuracy = None
    f1_macro = None
    for i, line in enumerate(lines):
        if "accuracy" in line.lower():
            try:
                accuracy = float(re.findall(r"\d+\.\d+", line)[0])
            except:
                pass
        if "macro avg" in line.lower():
            try:
                f1_macro = float(re.findall(r"\d+\.\d+", line)[2])  # f1-score est dans la3e colonne
            except:
                pass
    return accuracy, f1_macro

# extraction des résultats
resultats = []
for nom, fichier in fichiers.items():
    path = os.path.join(fichier)
    accuracy, f1_macro = extraire_scores(path)
    type_modele = "LogReg" if "LogReg" in nom else "BERT"
    resultats.append({
        "Modèle": nom,
        "Type": type_modele,
        "Accuracy": accuracy,
        "F1 macro": f1_macro
    })

df = pd.DataFrame(resultats)

# sauvegarde en CSV
df.to_csv(os.path.join("LG_scores_modeles.csv"), index=False)

# === Graphiques comparatifs globaux ===
# graphique 1 : accuracy
plt.figure(figsize=(10, 5))
colors = ['#1f77b4' if "LogReg" in m else '#ff7f0e' for m in df["Modèle"]]
df.plot(x="Modèle", y="Accuracy", kind="bar", legend=False, color=colors)
plt.title("Comparaison de l'Accuracy sur l'ensemble de test")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.xlabel("Modèles")
plt.tight_layout()
plt.savefig(os.path.join("accuracy_comparaison.png"))
plt.clf()

# graphique 2 : F1 macro
df.plot(x="Modèle", y="F1 macro", kind="bar", legend=False, color=colors)
plt.title("Comparaison du F1-score macro sur l'ensemble de test")
plt.ylabel("F1 macro")
plt.xticks(rotation=30)
plt.xlabel("Modèles")
plt.tight_layout()
plt.savefig(os.path.join("f1_macro_comparaison.png"))
plt.clf()

# === Graphiques par type de modèle ===
# Filtrer LogReg
df_logreg = df[df["Type"] == "LogReg"]
df_bert = df[df["Type"] == "BERT"]

# LogReg
df_logreg.plot(x="Modèle", y="Accuracy", kind="bar", color="#1f77b4", legend=False)
plt.title("Accuracy - Logistic Regression")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("accuracy_logreg.png")
plt.clf()

df_logreg.plot(x="Modèle", y="F1 macro", kind="bar", color="#1f77b4", legend=False)
plt.title("F1-score macro - Logistic Regression")
plt.ylabel("F1-score (macro)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("f1_logreg.png")
plt.clf()

# BERT
df_bert.plot(x="Modèle", y="Accuracy", kind="bar", color="#ff7f0e", legend=False)
plt.title("Accuracy - BERT")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("accuracy_bert.png")
plt.clf()

df_bert.plot(x="Modèle", y="F1 macro", kind="bar", color="#ff7f0e", legend=False)
plt.title("F1-score macro - BERT")
plt.ylabel("F1-score (macro)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("f1_bert.png")
plt.clf()

