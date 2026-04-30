# Projet de Machine Learning : Détection des Maladies Cardiovasculaires

## 1. Description du sujet
Le projet porte sur le développement d'un modèle de classification capable de prédire la présence ou l'absence de maladies cardiovasculaires (MCV) chez un individu. Les MCV représentent la première cause de mortalité dans le monde, et un diagnostic précoce est crucial pour la prise en charge des patients. 

L'enjeu est d'utiliser des données cliniques courantes (mesures physiques, résultats d'analyses sanguines et habitudes de vie) pour identifier les profils à risque. Ce projet s'inscrit dans le domaine de la santé numérique et nécessite une attention particulière à la gestion des données réelles, souvent imparfaites.

## 2. Problématique
**Dans quelle mesure un modèle de Machine Learning peut-il prédire avec fiabilité le risque cardiovasculaire à partir de données cliniques non invasives et souvent bruitées ?**

Cette problématique soulève trois défis majeurs :
1. **La qualité des données :** Comment traiter les valeurs aberrantes (ex: pressions artérielles irréalistes) ?
2. **L'enrichissement des données :** Peut-on améliorer la précision en fusionnant des sources cliniques avec des données nutritionnelles et biologiques avancées ?
3. **La performance médicale :** Comment minimiser les faux négatifs (patients malades non détectés) ?

## 3. Description des Datasets

### A. Dataset Principal : `cardio_train.csv`
* **Description :** 70 000 enregistrements de patients avec 12 variables (âge, sexe, taille, poids, pression artérielle, cholestérol, glucose, tabac, alcool, activité physique).
* **Méthode de collecte :** Recueilli lors d'examens médicaux réels (Source : Kaggle).
* **Justification du choix :** Volume important permettant un entraînement robuste et équilibre quasi-parfait des classes (50% de cas positifs / 50% négatifs).

### B. Dataset d'Enrichissement : `Nhanes_cvd_raw.csv`
* **Description :** Données issues de l'enquête nationale NHANES (2017-2023) incluant des biomarqueurs avancés.
* **Méthode de collecte :** Examens physiques et biologiques standardisés (CDC - États-Unis) (source : Kaggle)
* **Justification du choix :** Apporte des variables critiques absentes du premier dataset comme la **Protéine C-réactive (marqueur d'inflammation)**, le **Sodium (sel)** et les **Graisses saturées** via une technique d'imputation par regroupement (Binning & Mapping) : les variables manquantes sont estimées en faisant la jointure entre les deux datasets sur des critères démographiques et cliniques communs (tranches d'âge, genre et niveau d'hypertension)

## 4. Objectifs Business et Machine Learning

### Objectifs
* **Objectif Business :** Créer un outil d'aide au pré-diagnostic non invasif pour orienter les patients vers des spécialistes avant que des complications graves ne surviennent.
* **Objectif Machine Learning :** Entraîner un classifieur performant (type Random Forest ou XGBoost) capable de surpasser les scores de risque traditionnels en intégrant des variables comportementales et biologiques enrichies.

### Évaluation (Métriques)
Le succès du modèle sera évalué selon les priorités suivantes :
* **Recall (Rappel) :** Métrique prioritaire (minimiser les faux négatifs pour ne rater aucun patient à risque).
* **F1-Score :** Pour équilibrer la précision et le rappel.
* **Courbe ROC-AUC :** Pour mesurer la capacité de séparation des classes du modèle.

## 5. Stratégie de Nettoyage et Feature Engineering
Pour coller au "dataset idéal", les transformations suivantes sont prévues :
* **Nettoyage :** Filtrage des pressions artérielles (`ap_hi` et `ap_lo`) sur des plages physiologiques (ex: 80-220 mmHg).
* **Feature Engineering :**
    * Calcul de l'**IMC** (Poids / Taille²).
    * Conversion de l'âge de jours en années.
    * Création d'une variable "Hypertension" par paliers médicaux.
* **Fusion Statistique :** Injection des moyennes de consommation de sodium et de taux de protéine C-réactive issues de NHANES basées sur le profil (Âge/Sexe/IMC) du patient de `cardio_train`.