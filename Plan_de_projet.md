## Plan de projet

### 1. **Notre objet et les données**

* **Objet：** comparer la distribution des sentiments des commentaires sur Amazon Books et Amazon Kindle
* **Méthodes：** utiliser les modèles de classification des sentiments (Logistique Regression et BERT)
* **Données：** [Dataset Amazon Reviews sur HuggingFace ](https://amazon-reviews-2023.github.io/)

---

### 2. **Prétraitement des données**

* Taille d'échantillon choisie : comme le dataset est assez volumineux, on choisit de ne garder que les 10000 aléatoires pour éviter de surcharger le modèle et notre ordinateur.  [Echantillon.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/Echantillon.py)
* On utilise python [clean.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/clean.py) pour nettoyer les données, et produire `clean_text`
* Ajout des étiquettes ：`positive`, `negative`, `neutre`（garder ou pas）par [label.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/label.py)
* On extrait les informations pertinentes pour réduire la taille des données par [extraction.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/extraction.py)
---

### 3. **Conversion et division des données**

* Changer le format des données de `.jsonl` à `.csv`
* Diviser les données en `train.csv`, `dev.csv`, `test.csv` [split_dataset.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/split_dataset.py)

  * `train.csv`（80%）
  * `dev.csv`（10%）
  * `test.csv`（10%）

---

### 4. **Modèle de classification**

* On utilise `TfidfVectorizer` + `LogisticRegression` vs `Bert`
* On supporte la commande line pour inclure ou non les neutres
* On évalue le modèle sur `dev` et `test`

#### Logistic Regression
Pour notre premier modèle, nous avons utilisé un classifieur basé sur une combinaison de `TF-IDF` et `LogisticRegression`, qui est une méthode simple et efficace pour la classification de textes. L'ensemble du traitement est dans le script [logreg.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/logreg.py), dont voici les parties principales : 

* Chargement des données :
```python
def load_data(file_path, avec_neutre=True):
    df = pd.read_csv(file_path)
    if not avec_neutre:
        df = df[df['label'] != 'neutre']
    textes = df['clean_text'].fillna("").tolist()
    labels = df['label'].tolist()
    return textes, labels
```
Cette fonction peut non seulement lire notre fichier csv prétraité (contient les colonnes `clean_text` et `label`), mais aussi permet d'exclure les données neutres si l'option `avec_neutre` est `False`.

* Vectorisation des données :
```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf = vectorizer.transform(X_dev)
X_test_tfidf = vectorizer.transform(X_test)
```
On transforme le texte en vecteurs numériques en utilisant `TfidfVectorizer`，car les modèles ne peuvent pas traiter de texte directement. Ave la méthode de `TF-IDF`, elle donne un score à chaque mot selon sa fréquence dans le texte (term frequency) et son importance dans tous les textes (inverse document frequency)

Noté que le vocabulaire est limité aux 5000 termes les plus fréquents, c'est pour limiter la dimensionnalité et les risques d'overfitting.

* Entraînement du modèle :

```python
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
```
Avant d'entraîner le modèle, il faut transformer les étiquettes en nombres par `LabelEncoder`, car le modèle ne peut pas traiter des chaînes de caractères.

Par exemple : positive devient 2, negative devient 0, neutre devient 1.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train_encoded)
```
Ensuite on entraîne le modèle avec les textes vectorisés (TF-IDF) et les étiquettes encodées.
Le paramètre ici `max_iter=1000` permet au modèle de faire assez d'essais d'apprentissage pour bien apprendre à partir des données.


* Évaluation :
```python
y_dev_pred = model.predict(X_dev_tfidf)
print(classification_report(y_dev_encoded, y_dev_pred, target_names=le.classes_))
print(confusion_matrix(y_dev_encoded, y_dev_pred))
```
On évalue d'abord sur le jey de validation (`dev`) puis sur le jeu de test (`test`).

* Enregistrement des resultats :
```python
fichier_sortie = "resultats_logreg_avec_neutre.txt" if avec_neutre else "resultats_logreg_sans_neutre.txt"
    with open(fichier_sortie, "w", encoding="utf-8") as f:
        f.write(resultats)
```
Les résultats sont enregistrés dans un fichier texte nommé en fonction de l'option `avec_neutre` ou non.

* Ligne de commande : 
Nous avons également écrit un argparser pour permettre d'inclure ou non les neutres dans le modèle de classification.
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle de classification de sentiments.")
    parser.add_argument("--train", type=str, required=True, help="Chemin vers le fichier CSV d'entraînement")
    parser.add_argument("--dev", type=str, required=True, help="Chemin vers le fichier CSV de validation")
    parser.add_argument("--test", type=str, required=True, help="Chemin vers le fichier CSV de test")
    parser.add_argument("--avec-neutre", action="store_true", help="Inclure les commentaires neutres (label == 'neutre')")

    args = parser.parse_args()

    entrainer_et_evaluer(args.train, args.dev, args.test, avec_neutre=args.avec_neutre)
```
Exemple d'utilisation :
```
python logreg.py --train Kindle_3_train.csv  --dev Kindle_3_dev.csv --test Kindle_3_test.csv --avec-neutre
```

#### BERT

---

### 5. **Visualisation des résultats**

#### Logistic Regression
Afin de comparer plus visiblement les performances des modèles entraînés, nous avons écrit un script [visualisation.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/visualisation.py) qui lit automatiquement les résultats des fichiers texte, extrait les métriques accuracy et f1-score macro, puis génère des graphiques clairs pour faciliter notre analyse.

Voici un peu d'extrait du script :
```python
def extraire_scores(filepath): # on définit une fonction vers le chemin de fichier
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # initialisation des variables vides pour stocker les resultats
    accuracy = None
    f1_macro = None
    # parcours chaque ligne
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
```
On utilise le regex pour trouver les lignes contennat `accuracy` et `macro avg` et puis retourne les deux scores extraits.

```python
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
```
On applique la fonction `extraire_scores` à chaque fichier de résultats, et on stocke les résultats dans une dictionnaire `resultats`. Enfin on transforme la liste en dataframe pour la visualisation.

Voici un aperçu des résultats extraits automatiquement depuis les fichiers de test :

| Modèle             | Accuracy | F1 macro |
| ------------------ | -------- | -------- |
| Books (2 classes)  | 0.92     | 0.56     |
| Books (3 classes)  | 0.86     | 0.43     |
| Kindle (2 classes) | 0.94     | 0.54     |
| Kindle (3 classes) | 0.88     | 0.47     |

Et finalement, en utilisant `matplotlib`, on génère les graphiques suivants :

* [LR_accuracy_test.png](https://github.com/Xiaobo33/Extraction_info/blob/main/resultats/LR_accuracy_test.png)
* [LR_f1_macro_test.png](https://github.com/Xiaobo33/Extraction_info/blob/main/resultats/LR_f1_macro_test.png)

---





#### BERT
---

### 6. **Analyse des données**

* comparer les données Books et Kindle
* analyser la distribution des sentiments, les mots les plus fréquents, le modèle de classification
* comparer les résultats de Books et Kindle
* des limites et des perspectives