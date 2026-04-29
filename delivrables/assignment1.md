# Projet de Machine Learning : Détection des Maladies Cardiovasculaires

## 1. Description du sujet

Le projet porte sur le développement d'un modèle de classification capable de prédire la présence ou l'absence de maladies cardiovasculaires (MCV) chez un individu. Les MCV représentent la première cause de mortalité dans le monde, et un diagnostic précoce est crucial pour la prise en charge des patients.

L'enjeu est d'utiliser des données cliniques courantes (mesures physiques, résultats d'analyses sanguines et habitudes de vie) pour identifier les profils à risque. Ce projet s'inscrit dans le domaine de la santé numérique et nécessite une attention particulière à la précision des prédictions (pour éviter les faux négatifs) et à l'interprétabilité du modèle.

## 2. Première idée de problématique

**Problématique :** *Dans quelle mesure un modèle de Machine Learning peut-il prédire avec fiabilité le risque cardiovasculaire à partir de données cliniques non invasives et souvent bruitées ?*

Cette problématique soulève plusieurs défis :
* **La qualité des données :** Comment traiter les valeurs aberrantes (ex: pressions artérielles erronées) et les données manquantes ?
* **La sélection des variables :** Quels sont les indicateurs les plus déterminants (poids, cholestérol, pression artérielle) pour une prédiction robuste ?
* **La performance médicale :** Comment optimiser le rappel (recall) pour minimiser le risque de ne pas détecter un patient malade ?

## 3. À quoi ressemblerait un dataset idéal ?

Un dataset idéal pour ce projet ne se contenterait pas de quelques colonnes, il serait structuré de la manière suivante :

### A. Données cliniques et biétriques
* **Âge et Sexe :** Données démographiques de base.
* **Pression Artérielle (Ap_hi / Ap_lo) :** Mesures précises et répétées pour éviter les pics de stress ponctuel.
* **Indice de Masse Corporelle (IMC) :** Calculé à partir de la taille et du poids.

### B. Analyses biologiques (Laboratoire)
* **Bilan lipidique complet :** Pas seulement le cholestérol total, mais le détail LDL (mauvais cholestérol), HDL (bon cholestérol) et Triglycérides.
* **Glycémie à jeun :** Pour détecter le diabète, facteur de risque majeur.
* **Marqueurs inflammatoires :** Comme la Protéine C-réactive (CRP).

### C. Habitudes de vie et antécédents
* **Statut tabagique et consommation d'alcool :** Données quantifiées (nombre de cigarettes/verres par jour).
* **Niveau d'activité physique :** Mesuré idéalement via des capteurs (objets connectés) plutôt que déclaratif.
* **Historique familial :** Présence de maladies cardiaques chez les parents proches.

### D. Qualité technique
* **Équilibrage :** Un nombre égal de cas positifs et négatifs pour éviter les biais.
* **Volume :** Plus de 100 000 entrées pour permettre l'usage de modèles complexes.
* **Labels vérifiés :** Un diagnostic final confirmé par des examens de référence (ex: coronarographie ou IRM cardiaque).