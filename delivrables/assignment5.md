# Assignment 5 — Application Streamlit : CardioCheck ML PoC

## 1. Description de l'application

**CardioCheck ML PoC** est une application web interactive développée avec Streamlit.
Elle présente l'ensemble du projet de machine learning de détection du risque cardiovasculaire :
du contexte métier jusqu'à la démonstration live du modèle prédictif.

---

## 2. Objectif de l'interface

L'interface poursuit trois objectifs complémentaires :

- **Exploration** : expliquer la problématique métier et le pipeline de preprocessing à un public non technique.
- **Évaluation** : présenter les choix de modélisation et comparer les performances des modèles testés.
- **Prédiction** : permettre une démonstration interactive du modèle XGBoost sur un patient fictif.

---

## 3. Fonctionnalités implémentées

L'application est structurée en **3 pages** accessibles via une barre de navigation latérale.

### Page 1 — Objectifs & Preprocessing
- Onglet **Cadrage Business** : description de la problématique et de la valeur ajoutée de la fusion NHANES.
- Onglet **Méthode de Preprocessing** : description du pipeline de nettoyage (outliers, conversion d'unités) et du feature engineering (IMC, encodage OHE).

### Page 2 — Modélisation & Choix
- Justification du choix de XGBoost face à la Régression Logistique et au Random Forest.
- Tableau interactif des métriques des 3 modèles avec mise en évidence du meilleur recall (`model_metrics.csv`).
- Visualisations côte à côte :
  - Matrice de Confusion
  - Distribution du statut cardiaque par âge
  - Importance des variables XGBoost

### Page 3 — Démonstration Interactive
- Formulaire de saisie des données cliniques d'un patient.
- Calcul automatique de l'IMC à partir du poids et de la taille.
- Appel au modèle XGBoost et affichage du résultat avec probabilité de risque.

---

## 4. Inputs utilisateurs (Page Démonstration)

| Input | Type | Plage / Valeurs |
|---|---|---|
| Âge | Nombre entier | 18 – 100 ans |
| Poids | Nombre entier | 30 – 200 kg |
| Taille | Nombre entier | 130 – 220 cm |
| Sexe | Sélection | Femme / Homme |
| Tension Systolique (ap_hi) | Slider | 80 – 220 mmHg |
| Tension Diastolique (ap_lo) | Slider | 50 – 120 mmHg |
| Cholestérol | Sélection | Normal / Élevé / Très élevé |
| Glucose | Sélection | Normal / Élevé / Très élevé |
| Fumeur | Case à cocher | Oui / Non |
| Consommation d'alcool | Case à cocher | Oui / Non |
| Activité physique régulière | Case à cocher | Oui / Non |

> L'IMC est calculé automatiquement depuis le poids et la taille.
> Les variables Sodium et Saturated_Fat sont fixées à la valeur moyenne nationale issue de NHANES (2300 mg et 26 g respectivement), car ces données nutritionnelles ne sont pas disponibles en consultation clinique standard. Leur impact sur la prédiction est marginal au regard des variables dominantes (pression artérielle, âge, cholestérol).

---

## 5. Outputs affichés

### Page Modélisation
- **Tableau de métriques** : accuracy, precision, recall, f1 des 3 modèles avec surlignage du meilleur recall.
- **Matrice de confusion** : image `plot/matrice.png`.
- **Graphique âge/cardio** : image `plot/statut_cardiaque_age.png`.
- **Feature importance** : image `plot/variables_importantes_XGBoost.png`.

### Page Démonstration
- **Risque détecté** (rouge) : affiché si le modèle prédit `cardio = 1`, avec la probabilité en %.
- **Risque faible** (vert) : affiché si le modèle prédit `cardio = 0`, avec la probabilité en %.

---

## 6. Structure de l'application

Le fichier principal est `src/app.py`. Il est organisé en 4 fonctions :

| Fonction | Rôle |
|---|---|
| `build_app()` | Point d'entrée Streamlit, gère la navigation via la sidebar |
| `show_eda_and_preprocessing()` | Affiche la page Objectifs & Preprocessing |
| `show_model_logic()` | Affiche la page Modélisation & Choix |
| `show_demo_page()` | Affiche la page Démonstration Interactive |

ml-poc-project/
├── src/
│   ├── app.py          ← Application Streamlit (point d'entrée)
│   ├── config.py       ← Chemins centralisés du projet
│   ├── data.py         ← Chargement et split du dataset
│   └── features.py     ← Preprocessing et feature engineering
├── models/
│   └── xgboost_optimized.pkl  ← Modèle XGBoost entraîné
├── plot/
│   ├── matrice.png
│   ├── statut_cardiaque_age.png
│   └── variables_importantes_XGBoost.png
└── results/
├── model_metrics.csv
└── test_results.csv



---

## 7. Lancer l'application

### Prérequis
Être à la racine du projet avec le virtual environment activé :

```powershell
.venv\Scripts\Activate.ps1
Lancement

streamlit run src/app.py