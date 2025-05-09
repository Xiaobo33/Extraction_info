import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# lire les fichiers de résultats
fichiers = {
    "Books (2 classes)": "Books_logreg_sans_neutre.txt",
    "Books (3 classes)": "Books_logreg_avec_neutre.txt",
    "Kindle (2 classes)": "Kindle_logreg_sans_neutre.txt",
    "Kindle (3 classes)": "Kindle_logreg_avec_neutre.txt",
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
    resultats.append({
        "Modèle": nom,
        "Accuracy": accuracy,
        "F1 macro": f1_macro
    })

df = pd.DataFrame(resultats)

# sauvegarde en CSV
df.to_csv(os.path.join("LG_scores_modeles.csv"), index=False)

# graphique 1 : accuracy
df.plot(x="Modèle", y="Accuracy", kind="bar", legend=False, color="steelblue")
plt.title("Accuracy sur l'ensemble de test")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.xlabel("Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join("LR_accuracy_test.png"))
plt.clf()

# graphique 2 : F1 macro
df.plot(x="Modèle", y="F1 macro", kind="bar", legend=False, color="darkorange")
plt.title("F1-score macro sur l'ensemble de test")
plt.ylabel("F1 macro")
plt.xticks(rotation=30)
plt.xlabel("Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join("LR_f1_macro_test.png"))

print("C'est BON ! Graphiques générés et sauvegardés dans le dossier resultats/")
