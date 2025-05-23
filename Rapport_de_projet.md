## Rapport de projet : Comparaison des sentiments des commentaires sur deux corpus Amazon
### 1. **Introduction**

Notre projet vise à comparer la distribution des sentiments dans les avis clients publiés sur Amazon, en se concentrant sur deux catégories de produits culturels : `Books` et `Kindle`.

#### Problématique et objets du projet
La problématique est que : 
* Comment varie la distribution des sentiments dans les avis d'Amazon entre deux catégories de produits similaires (livres numériques et papier)?
* La classification binaire et la classification à trois classes ont-elles un impact significatif sur les performances des modèles ?

Donc le projet a trois objectifs principaux : 
* Effectuer une analyse de sentiments univarié sur les avis clients d'Amazon en utilisant deux approches : régression logistique et BERT.
* Comparer les résultats des approches dans deux situations différentes : avec et sans la classe `neutre`.
* Réaliser une extraction des NER dans les deux sous-corpus (Books et Kindle) pour mieux comprendre les sentiments exprimés par les clients selon la catégorie.

#### Données utilisées

Les données sont issues du corpus [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), disponible sur HuggingFace. Il s'agit d'un corpus massif, avec plusieurs catégories, contenant des millions d'avis vérifiés. Nous avons extrait et nettoyé des échantillons pour les catégories `Books` et `Kindle`.

---

### 2. **État de l'art**
L'analyse de sentiments est une tâche classique en TAL, notamment aux avis clients, elle est de plus en plus important pour les entreprises. Elle vise à classifier automatiquement les opinions exprimée par les clients en positifs, négatifs ou neutres. Nous avons étudié 6 articles sur l'analyse de sentiments surtout concentrés sur les avis Amazon.

Plusieurs études ont utilisé des méthodes d'apprentissage supervisé classique pour analyser les avis clients : 

On constate que les modèles les plus utilisés sont les modèles de régression logistique, Naïve Bayes, SVM et l'arbre de décision. Souvant avec l'aide des techniques de vectorisation simples comme TF-IDF ou Bow.

En ce qui concerne la classe neutre, la plupart des articles soulignent que l'ajout de cette classe rend la performance des modèles moins robustes. Certains auteurs donc proposent de supprimer cette classe pour améliorer la précision globale. En revanche, d'autres choisissent de la garder pour mieux refléter la diversité réelle des sentiments au prix d'une baisse du F1-score.

Plus récemment, des approches de deep learning ont été développées, notamment BERT et LSTM. Ces modèles sont capables de mieux capturer des informations complexes. Dans les contextes de multilingue, mBERT permet de traiter des avis dans plusieurs langues, mais il faut faire attention au transfert entre les langues, qui reste encore un défi.

L'ensemble des articles soulignent également sur l'importance du prétraitement, et quand au rôle du choix de vecteurs, TF-IDF est globalement plus performant que CountVectorizer. Pour les choix des modèles, ça dépend de différents objectif, par exemple pour un prototype rapide, un modèle simple peut suffire, mais pour produire un résultat robuste et précise, on utilise plutôt un modèle profond. Enfin l'accuacy ne suffit pas pour évaluer un modèle de sentiment, il faut considérer également le F1-score ou la matrice de confusion pour mesurer l'équilibre entre classes.

En conclusion, les travaux existants nous montrent une évolution claire des approches vers des modèles de plus en plus puissants pour adapter aux défis des langues. Les méthodes classique restent utile et pertinent pour les comparaisons de base, mais pour capter les nuances, les modèles comme BERT est plus efficaces.

L'état de l'art nous propose des idées pour le projet, nous choisissons donc un modèle classique et un modèle de deep learning pour comparer les performances.

---

### 3. **Prétraitement des données**

* Taille d'échantillon choisie : comme le dataset est assez volumineux, on choisit de ne garder que les 10000 aléatoires pour éviter de surcharger le modèle et notre ordinateur.  [Echantillon.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/Echantillon.py)
* On utilise python [clean.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/clean.py) pour nettoyer les données, et produire `clean_text`
* Ajout des étiquettes ：`positive`, `negative`, `neutre`（garder ou pas）par [label.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/label.py)
* On extrait les informations pertinentes pour réduire la taille des données par [extraction.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/extraction.py)
* Changer le format des données de `.jsonl` à `.csv`
* Diviser les données en `train.csv`, `dev.csv`, `test.csv` [split_dataset.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/split_dataset.py)

  * `train.csv`（80%）
  * `dev.csv`（10%）
  * `test.csv`（10%）

---

### 4. **Modèle de classification**

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
On transforme le texte en vecteurs numériques en utilisant `TfidfVectorizer`，car les modèles ne peuvent pas traiter de texte directement. Avec la méthode de `TF-IDF`, elle donne un score à chaque mot selon sa fréquence dans le texte (term frequency) et son importance dans tous les textes (inverse document frequency)

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
On évalue d'abord sur le jeu de validation (`dev`) puis sur le jeu de test (`test`).

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
L’autre modèle que nous avons utilisé repose sur l’architecture BERT. Nous avons choisi une version allégée proposée sur Hugging Face, **DistilBERT**, qui est pré-entraîné et bien adapté à la classification de texte. Le script [bert_train.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/bert_train.py) est utilisé pour entraîner et évaluer le modèle de manière automatisée.

*`load_csv_as_dataset()`*

Cette fonction permet de charger les trois fichiers (train / dev / test) nettoyés, puis de les convertir en objets Dataset compatibles avec la bibliothèque Trainer de Hugging Face. 

```python
train_dataset = Dataset.from_pandas(train_df[['clean_text', 'labels']])
```

Deux étapes importantes sont intégrées à cette fonction :

- D’une part, on supprime les exemples étiquetés "neutre" si l'option `--avec-neutre` n’est pas activée, afin de ne garder que deux classes.

```python
if not avec_neutre:
    train_df = train_df[train_df['label'] != 'neutre']
```

- D’autre part, les étiquettes textuelles (positive, negative, neutre) sont encodées en entiers à l’aide de `LabelEncoder` pour que le modèle puisse les interpréter correctement au moment de l’entraînement.

```python
train_df['labels'] = label_encoder.fit_transform(train_df['label'])
```

Une fois le `Dataset` préparé, chaque texte est **tokenizé** à l’aide du tokenizer associé à DistilBERT, afin d’être transformé en une séquence d’`input_ids` et de `attention_mask`, comme requis par les modèles BERT. Cette approche peut capturer non seulement la présence des mots, mais aussi leur **contexte dans la phrase**, ce qui représente un avantage clé par rapport à la vectorisation traditionnelle.

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = train_dataset.map(tokenize, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

*`compute_metrics()`*  

Pour évaluer les performances du modèle, nous avons défini cette fonction pour calculer automatiquement l'**Accuracy** et le **F1-score macro**. Ce dernier est très utile dans le cas de classes déséquilibrées. 

```python
def compute_metrics(eval_pred):
		logits, labels = eval_pred
		preds = np.argmax(logits, axis=1)
```

Les prédictions sont obtenues à partir des logits générés par le modèle, en appliquant la fonction `argmax`. Ces métriques sont ensuite utilisées automatiquement par l’objet `Trainer` pendant l’entraînement et l’évaluation.

*`Trainer` → Entraînement*

Pour entraîner le modèle, nous avons utilisé la classe `Trainer` de Hugging Face. Les paramètres d’entraînement ont été optimisés pour un bon compromis entre **qualité du résultat** et **temps de calcul** (car l’entraînement s’est fait en local sur CPU) :

```python
training_args = TrainingArguments(
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    num_train_epochs=2,
    max_steps=800,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="none"
)
```

Nous avons choisi 2 époques (au lieu de 3 initialement prévus), avec un batch size réduit, pour garantir un entraînement complet en moins de 15 minutes par modèle.

À l’aide de l’objet `Trainer` , les performances du modèle sont automatiquement évaluées sur le jeu de validation (dev) à la fin de chaque époque.

*`predict()` → Évaluation*

Après entraînement, le modèle est évalué à la fois sur le jeu de validation pour suivre la performance à chaque époque et sur le jeu de test (test) pour générer les résultats finaux. Nous transformons ensuite ces logits en classes prédictes (`preds`) à l’aide d’un `argmax`, et les comparons aux étiquettes réelles afin d’obtenir les métriques classiques : **precision**, **recall**, **f1-score**, et la **matrice de confusion**.

```python
dev_predictions = trainer.predict(dev_dataset)
y_dev_pred = np.argmax(dev_predictions.predictions, axis=1)
y_dev_true = dev_dataset["labels"]
```

Les résultats sont affichés et sauvegardés dans un fichier de texte spécifique à chaque corpus (par exemple : Books_bert_sans_neutre.txt) 

```python
with open(result_file, "w", encoding="utf-8") as f:
    f.write("[DEV] Résultats de validation :\n")
    f.write(dev_report)
```

*`argparse`*

Nos deux modèles utilisent la même méthode pour charger les jeux de données, à savoir un argument parser. Cela nous permet de spécifier facilement les fichiers d'entraînement, de validation et de test directement depuis la ligne de commande, et de relancer les expériences avec différentes configurations (avec ou sans la classe "neutre").

**Exemple d'utilisation :**

```bash
python bert_train.py --train Kindle_3_train.csv  --dev Kindle_3_dev.csv --test Kindle_3_test.csv --avec-neutre
```
---

### 5. **Visualisation des résultats**

#### Analyse quantitative
Afin de comparer plus visiblement les performances des modèles entraînés, nous avons écrit un script [visualisation.py](https://github.com/Xiaobo33/Extraction_info/blob/main/src/visualisation.py) qui lit automatiquement les résultats des fichiers texte, extrait les métriques **accuracy** et **f1-score macro**, puis génère des graphiques clairs pour faciliter notre analyse comparative.

Voici un extrait clé du script :
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

Voici un aperçu des résultats extraits automatiquement depuis les fichiers de texte :

| Modèle                | Accuracy | F1 macro |
|-----------------------|----------|----------|
| Books (2) - LogReg    | 0.92     | 0.56     |
| Books (2) - BERT      | 0.94     | 0.80     |
| Books (3) - LogReg    | 0.86     | 0.43     |
| Books (3) - BERT      | 0.87     | 0.51     |
| Kindle (2) - LogReg   | 0.94     | 0.54     |
| Kindle (2) - BERT     | 0.95     | 0.79     |
| Kindle (3) - LogReg   | 0.88     | 0.47     |
| Kindle (3) - BERT     | 0.89     | 0.56     |

Nous avons choisi d’extraire uniquement **l’accuracy** et le **F1-score macro**, car ces deux métriques permettent de juger à la fois la performance globale (accuracy) et la robustesse face au déséquilibre des classes (f1-macro). D’autres métriques comme la précision ou le rappel par classe sont également présentes dans les fichiers texte, mais nous avons préféré concentrer notre analyse visuelle sur ces deux valeurs les plus représentatives.

Et finalement, en utilisant `matplotlib`, on génère les graphiques suivants pour comparer clairement les performances des modèles sur l’ensemble des jeux de test : 
* ![accuracy_comparaison.png](https://github.com/Xiaobo33/Extraction_info/raw/main/resultats/accuracy_comparaison.png)
* ![f1_macro_comparaison.png](https://github.com/Xiaobo33/Extraction_info/raw/main/resultats/f1_macro_comparaison.png)

Résultats enregistrés dans ces fichiers de texte :  
* [Books_logreg_sans_neutre.txt](https://github.com/Xiaobo33/Extraction_info/blob/main/resultats/Books_logreg_sans_neutre.txt)
* [Books_logreg_avec_neutre.txt](https://github.com/Xiaobo33/Extraction_info/blob/main/resultats/Books_logreg_avec_neutre.txt)
* [Kindle_logreg_sans_neutre.txt](https://github.com/Xiaobo33/Extraction_info/blob/main/resultats/Kindle_logreg_sans_neutre.txt)
* [Kindle_logreg_sans_neutre.txt](https://github.com/Xiaobo33/Extraction_info/blob/main/resultats/Kindle_logreg_sans_neutre.txt)
* [Books_bert_sans_neutre.txt](https://github.com/Xiaobo33/Extraction_info/blob/main/resultats/Books_logreg_sans_neutre.txt)
* [Books_bert_avec_neutre.txt](https://github.com/Xiaobo33/Extraction_info/blob/main/resultats/Books_logreg_avec_neutre.txt)
* [Kindle_bert_sans_neutre.txt](https://github.com/Xiaobo33/Extraction_info/blob/main/resultats/Kindle_logreg_sans_neutre.txt)
* [Kindle_bert_sans_neutre.txt](https://github.com/Xiaobo33/Extraction_info/blob/main/resultats/Kindle_logreg_sans_neutre.txt)

---

#### Analyse lexicale
Afin de compléter l'analyse quantitative basée sur les scores de classification, nous avons généré des **nuages de mots** à partir des **avis positifs** issus des corpus *Books* et *Kindle*. Dans les deux cas, les mots dominants sont book, read, story, character, love, ce qui témoigne d’une expérience de lecture immersive et émotionnelle largement partagée par les utilisateurs.

* ![books_wordcloud.png](https://github.com/Xiaobo33/Extraction_info/raw/main/resultats/books_wordcloud.png)

* ![kindle_wordcloud.png](https://github.com/Xiaobo33/Extraction_info/raw/main/resultats/kindle_wordcloud.png)

On note cependant que le corpus Kindle fait légèrement ressortir des expressions à tonalité plus fonctionnelle, telles que “*reader”*, “*next book”*, “*can’t wait”*, “*look forward”*, ce qui pourrait refléter une approche plus utilitaire ou séquentielle des **lectures numériques**.

En fait, malgré l'utilisation de deux sous-corpus distincts, les nuages de mots présentent une **forte similarité lexicale**. Ce phénomène s'explique par deux facteurs principales : 

- Les commentaires positifs, qu’ils portent sur un livre papier ou numérique, se concentrent souvent sur les mêmes éléments d’appréciation comme qualité de l’histoire, attachement aux personnages, style de l’auteur, etc.
- Les utilisateurs d’Amazon tendent à adopter un **style d’évaluation très homogène**, souvent court, enthousiaste, et centré sur la recommandation, avec des formules récurrentes telles que “loved it”, “highly recommend”, “couldn’t put it down”.

Cette observation suggère que, dans le cas de la classification de sentiments, **le type de support** (papier ou numérique) n’influence pas de manière significative le langage employé dans les évaluations favorables.

---

### 6. **Analyse des résultats**

#### Les performances de Logistic Regression

Les modèles de régression logistique présentent des performances globalement correctes, surtout dans les tâches de classification binaire. Sur les corpus Books et Kindle, les modèles entraînés sans la classe neutre atteignent une accuracy de **94%**, et un F1-score macro d'environ **0.54 à 0.56**, ce qui traduit une forte capacité du modèle à reconnaître les avis positifs, mais une difficulté à détecter les avis négatifs.

Lorsque l'on introduit la classe neutre, la performance du modèle diminue visiblement. L'accuracy chute à **87%**, et le F1 macro descend jusqu'à **0.43 à 0.47**. Cela met en évidence la difficulté du modèle régression logistique à traiter des tâches multi-classes plus nuancées.

En ce qui concerne l'analyse des matrices de confusion et des scores par classe permet de mieux comprendre cette baisse. Dans les deux corpus, la classe positive reste toujours très bien prédite (rappel ≥ 97%, f1-score ≥ 0.93). La classe negative bénéficie d'une amélioration par rapport à la version binaire, avec un rappel qui passe de 6–9% à 24–65%. En revanche, la classe neutre est systématiquement mal prédite : le modèle n'identifie presque aucun exemple de cette classe (aucun exemple dans le corpus de Books) avec un rappel inférieur à 10 %.

En conclusion, la régression logistique est un bon choix pour la classification binaire. Elle est capable de distinguer les avis positifs et négatifs, mais elle ne distingue pas la classe neutre. Ces résultats soulignent l'intérêt de recourir à des modèles plus puissants adaptées pour améliorer la reconnaissance des classes minoritaires.

---

#### Les performances de BERT

Les modèles BERT ont affiché dans l'ensemble de **meilleures performances** que la régression logistique, en particulier pour les versions à **deux classes** (positive / negative). Pour les corpus *Books* et *Kindle*, on observe une accuracy **supérieure à 94%** et un F1 macro **proche de 0.80**, ce qui montre une bonne capacité du modèle à capturer les relations contextuelles dans les textes.

Cependant, lorsque la tâche devient une classification à **trois classes** (positive, neutre, negative), la performance globale baisse : l'accuracy descend autour de **87-89%**, et le F1-score macro tombe à **0.51–0.56**. 

Après avoir comparé d'autres métriques, on trouve que cette chute peut s'expliquer par deux facteurs. D'une part, la classe neutre est sous-représentée dans les données, ce qui rend l'apprentissage plus difficile. D'autre part, la frontière sémantique entre "neutre" et "positive" est parfois floue, surtout dans des commentaires courts ou ambigus. 

On constate notamment dans les fichiers de sortie que la classe neutre est très **mal prédite** par le modèle : son **recall est inférieur à 10%**, ce qui signifie qu'elle est très souvent confondue avec la classe positive. Ce phénomène est visible également dans les matrices de confusion où une majorité de commentaires neutres sont mal classés.

D'un point de vue pratique, cela signifie que même si le modèle reste très performant sur les cas clairs (positifs/négatifs), il a des difficultés à détecter des nuances plus faibles. 

Comme le présente la cinquième partie *Visualisation des résultats*, nous avons choisi de nous concentrer sur deux métriques représentatives pour faire une comparaison entre les deux modèles. Néanmoins, pour une analyse plus fine, la précision et le rappel par classe restent très utiles pour comprendre où le modèle se trompe, notamment en analysant les erreurs sur la classe neutre. 

Ainsi, malgré la performance générale très satisfaisante des modèles BERT, une attention particulière doit être portée à la gestion des classes minoritaires et à l'interprétation des cas ambigus.

---

#### Conclusion partielle (Régression Logistique vs. BERT)

1. Classification binaire
Les deux modèles ont des performances similaires sur les corpus Books et Kindle, avec une accuracy superieure à 94%. Toutefois, leurs F1-score varient, avec la régression logistique ayant un F1 macro proche de 0.54 à 0.56, tandis que le BERT a un F1 macro proche de 0.80. Cela montre que BERT généralise mieux aux deux classes.

2. Classification à trois classes
Quand on introduit la classe neutre, la performance des deux modèles diminue, mais pas dans les mêmes proportions. L'accuracy chute à environ 87% pour la régression logistique, et à 87–89% pour BERT. Et le F1-score macro descend à 0.43–0.47 pour logreg, et reste légèrement supérieur avec BERT 0.51–0.56.

Cependant, cette amélioration ne résout pas le problème principal, qui est la difficulté à distinguer la classe neutre. Cela est particulièrement visible dans les matrices de confusion, où la majorité des commentaires neutres sont mal classés.

En conclusion, BERT dépasse globalement la régression logistique, mais aucun des deux modèles ne parvient à gérer efficacement la classe neutre sans ajustements spécifiques. Cela ouvre la voie à des perspectives d'amélioration.

---


### 7. Conclusion (des limites et des perspectives)

Ce projet nous a permis de mettre en œuvre deux approches différentes pour la tâche de classification des sentiments dans les avis Amazon, en comparant deux sous-corpus (Books et Kindle), et deux types de modèles : **logistic regression** et **BERT**.

En général, nous avons pu constater les tendances dans les articles scientifiques que nous avons étudiés : 
* Les modèles classique restent efficaces pour des tâches simples de classification binaire, grâce à leur rapidité et à leur interprétabilité.
* Les modèles de deep learning, qui est plus complexe, montrent de meilleures performances globales, notamment grâce à leur capacité à capter les nuances contextuelles du langage.

Pourtant, les deux modèles rencontrent des difficultés dès lors que la classe `neutre` est introduite. Les matrices de confusion montrent qu’elle est fréquemment confondue avec la classe `positive`, ce qui révèle à la fois un déséquilibre dans les données et une frontière sémantique floue entre les catégories.

Au cours de notre projet, nous avons été confrontés à plusieurs difficultés : 
* La taille des données était trop importante pour les machines de notre époque, et il est nécessaire de les réduire.
* Le déséquilibre entre les classes peut rendre l'entraînement moins stable, ici nous avons beaucoup moins de données pour la classe `neutre` et `négative`.
* La limite de nos ressources matérielles, ainsi que la complexité des modèles (notamment BERT) nous restreint le nombre possible pour l'entraînement.

Selon nos résultats, nous avons pensé quelques perspectives d'amélioration : 
* La pondération des classes peut être un moyen de réduire le déséquilibre entre les classes, en augmentant la contribution des classes minoritaires comme `neutre`.
* L'augmentation de données peut permettre d'améliorer la performance globale des modèles, mais besoin d'un ordinateur plus puissant.
* L'utilisation des modèles plus puissants, comme RoBERTa.
* L'évaluation humaine pour valider les prédictions et affiner les critères de la classification, mais cela est difficile à réaliser car le nombre de cas est important.
* La gestion des cas ambigus peut être un enjeu majeur, en particulier dans le cas de la classe `neutre`.

Bref, nous avons identifié plusieurs pistes d'amélioration, que nous pourrions explorer dans de futurs travaux pour renforcer la détection des sentiments les plus subtils, notamment la classe neutre.