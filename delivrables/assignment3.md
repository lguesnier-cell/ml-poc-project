# Assignment 3 : Modélisation et Évaluation

## 1. Définition du problème ML

Il s'agit d'un problème de **classification binaire supervisée** : prédire la présence ou l'absence de maladie cardiovasculaire (`cardio` = 1 ou 0) à partir de données cliniques et nutritionnelles.

- **Variable cible :** `cardio` (0 = absence, 1 = présence)
- **Features :** données démographiques, cliniques (pression artérielle, IMC) et nutritionnelles (Sodium, Saturated_Fat issus de NHANES)
- **Contexte médical :** le coût d'un faux négatif (patient malade non détecté) est bien supérieur au coût d'un faux positif (fausse alerte). La métrique d'évaluation est choisie en conséquence.

## 2. Métrique d'évaluation

La **métrique principale est le Recall (sensibilité)** :

$$Recall = \frac{TP}{TP + FN}$$

En contexte médical, minimiser les faux négatifs est prioritaire : un patient à risque non détecté représente un danger grave. 

Les métriques secondaires calculées pour comparaison complète sont : **accuracy**, **precision** et **F1-score**. Elles sont toutes implémentées dans `src/metrics.py` et sauvegardées dans `results/model_metrics.csv`.

## 3. Protocole d'évaluation

- **Split train/test :** 80% entraînement / 20% test, stratifié sur `cardio` (`random_state=42`) — implémenté dans `src/data.py`
- **Scaling :** `StandardScaler` appliqué après le split (`fit` sur `X_train`, `transform` sur `X_test`) pour éviter le data leakage
- **Évaluation finale :** chaque modèle est évalué sur `X_test` / `y_test` via `src/metrics.py`, les résultats sont comparés dans `results/model_metrics.csv`
- **Reproductibilité :** `random_state=42` fixé sur le split et les modèles qui l'acceptent

## 4. Les trois modèles sélectionnés

### Modèle 1 — Régression Logistique (`models/baseline_logreg.pkl`)

**Hypothèses principales**
- La relation entre les features et la probabilité d'avoir une MCV est approximativement linéaire.
- Les features sont indépendantes les unes des autres.

**Avantages attendus**
- Modèle de référence (baseline) simple et interprétable.
- Coefficients directement lisibles : permet d'identifier les features les plus discriminantes.
- Convergence rapide, robuste sur des données bien scalées.

**Limites attendues**
- Ne capture pas les interactions non-linéaires entre features (ex. : BMI × pression artérielle).
- Sensibilité potentiellement plus faible que les modèles ensemblistes.

**Adéquation avec le problème et la métrique**
Bonne adéquation comme baseline. Le seuil de décision peut être ajusté pour maximiser le Recall au détriment de la précision si nécessaire.

---

### Modèle 2 — Random Forest (`models/random_forest.joblib`)

**Hypothèses principales**
- Les relations entre features et cible sont non-linéaires et peuvent être capturées par des arbres de décision.
- La combinaison de nombreux arbres entraînés sur des sous-échantillons réduit la variance (bagging).

**Avantages attendus**
- Robuste aux outliers résiduels et aux features redondantes.
- Fournit une importance des variables (feature importance) utile pour l'interprétabilité clinique.
- Moins sensible au scaling que la régression logistique.

**Limites attendues**
- Tend à être plus conservateur sur la classe minoritaire si le dataset est déséquilibré.
- Moins performant que le boosting sur des données tabulaires de taille moyenne.

**Adéquation avec le problème et la métrique**
Bonne adéquation. Le paramètre `class_weight='balanced'` peut être activé pour améliorer le Recall sur la classe positive.

---

### Modèle 3 — XGBoost (`models/xgboost_optimized.pkl`)

**Hypothèses principales**
- Les résidus d'erreurs successifs peuvent être corrigés itérativement par des arbres additifs (boosting de gradient).
- Le dataset contient suffisamment d'exemples pour que le boosting converge sans sur-apprendre.

**Avantages attendus**
- Meilleure performance attendue sur données tabulaires de taille moyenne.
- Le paramètre `scale_pos_weight` permet de gérer directement le déséquilibre de classes en faveur du Recall.
- Régularisation intégrée (L1/L2) qui réduit le sur-apprentissage.

**Limites attendues**
- Plus complexe à interpréter qu'une régression logistique.
- Hyperparamètres nombreux — nécessite une optimisation (grid search ou random search) pour atteindre ses meilleures performances.

**Adéquation avec le problème et la métrique**
Très bonne adéquation. C'est le modèle principal du projet, retenu pour son équilibre entre performance et contrôle du déséquilibre de classes.

## 5. Justification du choix des trois modèles

Les trois modèles couvrent une progression logique en complexité :

| Modèle | Complexité | Interprétabilité | Recall attendu |
|---|---|---|---|
| Régression Logistique | Faible | Élevée | Moyen |
| Random Forest | Moyenne | Moyenne | Bon |
| XGBoost | Élevée | Faible | Meilleur |

Cette progression permet de vérifier que la complexité supplémentaire apporte un gain réel sur le Recall, et de conserver un modèle interprétable (régression logistique) comme référence pour valider la cohérence clinique des prédictions.

## 6. Notebooks

- `notebooks/modelling.ipynb` : contient l'entraînement des trois modèles, la comparaison des métriques et l'analyse des résultats.

**Pour reproduire les expériences :**
1. S'assurer que les données sont disponibles dans `data/raw/`
2. Exécuter `notebooks/preprocessing.ipynb` pour générer les datasets dans `data/processed/`
3. Exécuter `notebooks/modelling.ipynb`
4. Les modèles entraînés sont sauvegardés dans `models/`, les métriques dans `results/model_metrics.csv`