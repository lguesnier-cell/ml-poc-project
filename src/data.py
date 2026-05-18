import pandas as pd
import os
from sklearn.model_selection import train_test_split
from features import preprocess_cardio_data, apply_feature_engineering


def load_dataset_split():
    # On récupère le chemin du dossier racine du projet (un cran au-dessus de 'src')
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # On construit les chemins complets vers les fichiers
    path_cardio = os.path.join(base_path, "data", "raw", "cardio_train.csv")
    path_nhanes = os.path.join(base_path, "data", "raw", "Nhanes_cvd_raw.csv")
    
    # Chargement
    df_cardio = pd.read_csv(path_cardio, sep=';')
    df_nhanes = pd.read_csv(path_nhanes)

    # Étape 1 : Nettoyage initial[cite: 3, 4]
    df_cardio = preprocess_cardio_data(df_cardio)
    
    # Étape 2 : Préparation de la table NHANES pour la fusion statistique[cite: 2, 5]
    # On reproduit la logique d'agrégation basée sur l'âge et l'IMC[cite: 2]
    age_bins = [0, 30, 35, 40, 45, 50, 55, 60, 65, 100]
    age_labels = ['0-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65+']
    df_nhanes['age_group'] = pd.cut(df_nhanes['Age'], bins=age_bins, labels=age_labels)
    
    # On définit les catégories d'IMC pour NHANES
    df_nhanes['bmi_cat'] = pd.cut(df_nhanes['BMI'], bins=[0, 18.5, 25, 30, 100], 
                                  labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    ref_table = df_nhanes.groupby(['age_group', 'bmi_cat'], observed=True)[['Sodium', 'Saturated_Fat']].mean().reset_index()

    # Étape 3 : Feature Engineering complet et Fusion[cite: 3, 6]
    df_final = apply_feature_engineering(df_cardio, ref_table)

    # Étape 4 : Séparation et Split avec stratification[cite: 5]


    X = df_final.drop(columns=['cardio'])
    y = df_final['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



    return X_train, X_test, y_train, y_test