# Assignment 2 : Préparation des Données et Feature Engineering

## 1. Description des étapes de nettoyage des données

Le nettoyage a visé à éliminer les valeurs aberrantes et les bruits physiologiques pour garantir la robustesse des modèles :
- **Pressions artérielles :** Filtrage de la pression systolique (`ap_hi`) entre 80 et 220 mmHg et de la pression diastolique (`ap_lo`) entre 50 et 120 mmHg.
- **Paramètres corporels :** - Taille maintenue au-dessus de 140 cm.
    - Poids maintenu au-dessus de 40 kg.
    - IMC (BMI) filtré entre 15 et 60 pour exclure les cas extrêmes non représentatifs d'une étude de population générale.

## 2. Nouvelles features créées

Pour enrichir le dataset original, trois types de données ont été intégrés ou transformés :
- **Indice de Masse Corporelle (BMI) :** Calculé à partir du poids et de la taille ($poids / taille^2$) pour mieux représenter la corpulence que le poids seul.
- **Données NHANES (Imputation par fusion) :** Intégration de biomarqueurs externes (`Sodium`, `Saturated_Fat`) via une jointure statistique basée sur l'âge et l'IMC. Cela ajoute une dimension nutritionnelle cruciale au risque cardiovasculaire.
- **Age en années :** Conversion de l'âge (initialement en jours) en années pour faciliter l'interprétation clinique.

## 3. Transformations appliquées (Encoding & Scaling)

### Encodage (Encoding)
Pour rendre les données exploitables par tous les algorithmes (linéaires et basés sur les arbres) :
- **Variables Binaires :** `gender` a été transformé en format 0/1. Les variables `smoke`, `alco`, et `active` ont été conservées en format binaire.
- **One-Hot Encoding :** Appliqué sur `cholesterol` et `gluc` (3 niveaux chacun).
    - **Choix technique :** Utilisation de l'argument `drop_first=True`.

### Mise à l'échelle (Scaling)
- **Méthode retenue :** `StandardScaler` (Z-score normalization).
- **Stratégie :** Le scaling sera appliqué au moment du split train/test pour éviter le *data leakage*. 

## 4. Justification des choix et alternatives

### Gestion de la Multicollinéarité
Suite à l'analyse de la Heatmap de corrélation, nous avons fait les choix suivants :
- **Suppression de `weight` et `height` :** Car l'information est déjà synthétisée dans le `bmi`.
- **Suppression de `pulse_pressure` et `hypertension_stage` :** Ces variables créaient une redondance trop forte avec les pressions artérielles brutes, risquant de biaiser les coefficients des modèles linéaires.

### Alternatives testées et non retenues
- **Imputation par la moyenne :** Rejetée pour les données NHANES au profit d'une fusion par profil (âge/IMC) afin de conserver la variance naturelle des habitudes alimentaires.
- **Label Encoding pour le cholestérol :** Bien que plus simple, cette méthode a été rejetée car elle impose un ordre mathématique linéaire (1 < 2 < 3) qui peut induire le modèle en erreur sur le poids réel de chaque catégorie.

## 5. Impact attendu sur les modèles

- **Robustesse :** Le filtrage des outliers empêche les erreurs de mesure d'influencer négativement l'entraînement.
- **Précision du Rappel (Recall) :** L'ajout du `Sodium` et des graisses saturées devrait permettre au modèle de mieux capturer les profils à risque (Classe 1), minimisant ainsi les faux négatifs.
- **Convergence :** L'encodage One-Hot associé au Scaling futur permettra une convergence plus stable des algorithmes d'optimisation.

## 6. Structure du Repository et Utilisation

### Fichiers .py à modifier
- `src/data.py` : Contient la logique de chargement et de séparation des données (`load_dataset_split`).
- `src/features.py` : Définit la liste finale des features sélectionnées et les transformations finales.
- `src/metrics.py` : Préparé pour intégrer le calcul du Recall et du F1-score.

### Données et Notebooks
- **Notebooks :** - `notebooks/preprocessing_1.ipynb` : Script de nettoyage initial et fusion NHANES.
    - `notebooks/preprocessing_2.ipynb` : Script final d'encodage et export du dataset propre.
- **Datasets :** - `data/processed/df_clean.csv` : Données nettoyées sans encodage.
    - `data/processed/df_encoded.csv` : Dataset final utilisé pour le modelling.

**Chargement des données :**
```python
import pandas as pd
# Le dataset transformé est stocké dans le dossier processed
df = pd.read_csv('../data/processed/df_encoded.csv')