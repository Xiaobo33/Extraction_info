## Plan de projet

### 1. **Notre objet et les données**

* **Objet：** comparer la distribution des sentiments des commentaires sur Amazon Books et Amazon Kindle
* **Méthodes：** utiliser un modèle de classification des sentiments
* **Données：** [Dataset Amazon Reviews sur HuggingFace ](https://amazon-reviews-2023.github.io/)

---

### 2. **Prétraitement des données**

* Taille d'échantillon choisie : 10000 par [Echantillon.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/Echantillon.py)
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

---

### 5. **Visualisation des résultats**

* Comparer les résultats sur les datasets dev/test ：

  * accuracy
  * F1 score (weighted / macro)
* en utilisant `matplotlib`

---

### 6. **Analyse des données**

* comparer les données Books et Kindle
* analyser la distribution des sentiments, les mots les plus fréquents, le modèle de classification
* comparer les résultats de Books et Kindle
* des limites et des perspectives