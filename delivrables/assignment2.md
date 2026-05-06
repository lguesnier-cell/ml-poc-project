# Assignment 2 : Préparation et Transformation des Données

Ce document résume les étapes de nettoyage et de préparation du dataset cardiovasculaire combiné aux données NHANES.

## 1. Nettoyage des données
Le nettoyage a visé l'élimination des anomalies physiques et médicales :
*   **Filtrage physiologique** : Conservation des pressions artérielles systoliques (`ap_hi`) entre 80 et 220 et diastoliques (`ap_lo`) entre 50 et 120.
*   **Limites corporelles** : Conservation des IMC (BMI) entre 15 et 60, taille $\ge$ 140 cm et poids $\ge$ 40 kg[cite: 5, 6].
*   **Cohérence médicale** : Suppression des entrées où `ap_hi` $\le$ `ap_lo` (pression systolique toujours supérieure à la diastolique)[cite: 5].

## 2. Transformations et Nouvelles Features
*   **Fusion Statistique (NHANES)** : Ajout des variables `Sodium`, `C_Reactive` et `Saturated_Fat` par imputation des moyennes calculées par profil (Age + Catégorie d'IMC) dans NHANES.
*   **Feature Engineering** : 
    *   `BMI` : Calculé à partir du poids et de la taille.
    *   `Hypertension_stage` : Création d'une échelle clinique de 0 à 3 basée sur les paliers de pression artérielle.
    *   `Pulse_pressure` : Calcul de la différence entre pression systolique et diastolique.
*   **Scaling** : Application d'un `StandardScaler` sur les variables numériques pour normaliser les échelles.

## 3. Justification des choix et alternatives
*   **Variables supprimées** : Les caractéristiques `weight`, `C_reactive`, `pulse_pressure` et `hypertension_stage` ont été supprimées car elles présentaient une trop forte corrélation avec d'autres variables (ex: `weight` vs `BMI` ou `pulse_pressure` vs `ap_hi`). Leur maintien aurait induit de la **multicollinéarité**, nuisant à la stabilité des modèles.
*   **Alternative non retenue (PCA)** : L'Analyse en Composantes Principales (PCA) n'a pas été retenue afin de préserver l'**interprétabilité clinique** des résultats. La conservation des variables originales permet de quantifier précisément l'impact de facteurs réels (ex: cholestérol) sur le risque cardiaque, contrairement aux axes abstraits de la PCA.
*   **Précision du Split** : Le découpage Train/Test est effectué **avant** le scaling dans `src/data.py` pour éviter toute fuite de données.

## 4. Impact attendu
Le nettoyage des outliers et la normalisation des échelles par scaling devraient améliorer la convergence des modèles et éviter que des variables à forte amplitude (comme le Sodium) n'écrasent les autres prédicteurs.

## 5. Gestion des données dans le Repository
*   **Source** : Le fichier nettoyé `df_clean.csv` est généré par le notebook `preprocessing.ipynb`.
*   **Localisation** : Stocké dans `data/processed/`.
*   **Chargement** : Utiliser `src.data.load_dataset_split()` qui automatise le chargement et les transformations via `src/features.py`.