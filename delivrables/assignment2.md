# Assignment 2 : Préparation des Données et Feature Engineering

## 1. Description des étapes de nettoyage des données

Le nettoyage est implémenté dans `src/features.py` via la fonction `preprocess_cardio_data`. Il vise à éliminer les valeurs aberrantes et les incohérences physiologiques :

- **Pressions artérielles :** Filtrage de la pression systolique (`ap_hi`) entre 80 et 220 mmHg et de la pression diastolique (`ap_lo`) entre 50 et 120 mmHg. Une vérification de cohérence supplémentaire assure que `ap_hi > ap_lo`.
- **Paramètres corporels :**
    - Taille maintenue au-dessus de 140 cm.
    - Poids maintenu au-dessus de 40 kg.
    - IMC (BMI) filtré entre 15 et 60 pour exclure les cas extrêmes non représentatifs d'une étude de population générale.

## 2. Nouvelles features créées

Pour enrichir le dataset original, trois types de données ont été intégrés ou transformés (fonction `preprocess_cardio_data` et `apply_feature_engineering` dans `src/features.py`) :

- **Indice de Masse Corporelle (BMI) :** Calculé à partir du poids et de la taille (`poids / taille²`) pour mieux représenter la corpulence que le poids seul.
- **Age en années (`age_years`) :** Conversion de l'âge (initialement en jours) en années entières pour faciliter l'interprétation clinique et la jointure avec NHANES.
- **Données NHANES (Imputation par fusion) :** Intégration de biomarqueurs externes (`Sodium`, `Saturated_Fat`) via une jointure statistique basée sur l'âge et l'IMC. Des clés de jointure intermédiaires (`age_group`, `bmi_cat`) sont créées puis supprimées après la fusion.

## 3. Transformations appliquées (Encoding & Scaling)

### Encodage (Encoding)

Pour rendre les données exploitables par tous les algorithmes :

- **Variables binaires :** `gender` est en format 0/1. Les variables `smoke`, `alco`, et `active` sont conservées en format binaire natif.
- **One-Hot Encoding :** Appliqué sur `cholesterol` et `gluc` (3 niveaux chacun) via `pd.get_dummies` avec `drop_first=True` pour éviter la multicollinéarité parfaite.

### Mise à l'échelle (Scaling)

- **Méthode retenue :** `StandardScaler` (Z-score normalization).
- **Statut actuel :** Le scaling est appliqué dans `src/data.py` après le split train/test :
    - `fit_transform` sur `X_train` uniquement — le scaler apprend la moyenne et l'écart-type sur les données d'entraînement.
    - `transform` seul sur `X_test` — les mêmes paramètres sont appliqués sans observer les données de test, ce qui évite le *data leakage*.
    
## 4. Justification des choix et alternatives

### Gestion de la Multicollinéarité

Suite à l'analyse de la heatmap de corrélation, les choix suivants ont été faits (dans `apply_feature_engineering`) :

- **Suppression de `weight` et `height` :** L'information est déjà synthétisée dans le `bmi`.
- **Suppression de `age`** (en jours) : Remplacé par `age_years` plus lisible et interprétable.
- **Suppression de `id`** : Identifiant sans valeur prédictive.
- **Suppression de `age_group` et `bmi_cat`** : Clés de jointure intermédiaires, inutiles après la fusion.

### Alternatives testées et non retenues

- **Imputation par la moyenne globale :** Rejetée pour les données NHANES au profit d'une fusion par profil (âge/IMC) afin de conserver la variance naturelle des habitudes alimentaires selon les tranches de population.
- **Label Encoding pour `cholesterol` et `gluc` :** Bien que plus simple, cette méthode impose un ordre mathématique linéaire (1 < 2 < 3) qui peut induire le modèle en erreur sur le poids réel de chaque catégorie. L'OHE est préféré.
- **Variables dérivées supplémentaires (`pulse_pressure`, `hypertension_stage`) :** Explorées puis rejetées car elles créaient une redondance forte avec les pressions artérielles brutes, risquant de biaiser les coefficients des modèles linéaires.

## 5. Impact attendu sur les modèles

- **Robustesse :** Le filtrage des outliers et la vérification `ap_hi > ap_lo` empêchent les erreurs de mesure d'influencer négativement l'entraînement.
- **Précision du Rappel (Recall) :** L'ajout du `Sodium` et des graisses saturées devrait permettre au modèle de mieux capturer les profils à risque (Classe 1), minimisant les faux négatifs. Ce gain est mesurable via `metrics.py` qui calcule accuracy, precision, recall et f1.
- **Convergence :** L'encodage One-Hot associé au futur StandardScaler permettra une convergence plus stable des algorithmes d'optimisation (régression logistique, SVM).
- **Réduction du bruit :** La suppression des features redondantes réduit la dimensionnalité et évite d'introduire du bruit dans les modèles basés sur les distances.

## 6. Fichiers .py modifiés

### `src/data.py`

Orchestre le pipeline complet :
1. Charge `data/raw/cardio_train.csv` (séparateur `;`) et `data/raw/Nhanes_cvd_raw.csv`.
2. Appelle `preprocess_cardio_data` (nettoyage + création BMI/age_years).
3. Agrège les données NHANES par tranches d'âge et catégories IMC.
4. Appelle `apply_feature_engineering` (fusion NHANES, OHE, suppression des colonnes redondantes).
5. Retourne le split train/test stratifié (80/20, `random_state=42`).

### `src/metrics.py`

Implémente `compute_metrics(y_true, y_pred)` qui retourne un dictionnaire avec quatre métriques :

```python
{
    "accuracy": float,
    "precision": float,
    "recall": float,
    "f1": float,
}