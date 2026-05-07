import pandas as pd
import numpy as np

def get_bmi_cat(bmi):
    """Catégorisation de l'IMC selon les standards cliniques."""
    if bmi < 18.5: return 'Underweight'
    if bmi < 25: return 'Normal'
    if bmi < 30: return 'Overweight'
    return 'Obese'

def preprocess_cardio_data(df):
    """Nettoyage initial et feature engineering de base sur cardio_train[cite: 2, 4]."""
    # Filtrage des pressions artérielles (Plages physiologiques réalistes)[cite: 2, 4]
    df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 220)]
    df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 120)]
    df = df[df['ap_hi'] > df['ap_lo']] # Vérification de cohérence[cite: 4]
    
    # Filtrage Taille et Poids minimums
    df = df[df['height'] >= 140]
    df = df[df['weight'] >= 40]
    
    # Création de l'IMC et de l'âge en années[cite: 2, 3]
    df['age_years'] = (df['age'] / 365.25).astype(int)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Filtrage de l'IMC pour exclure les valeurs extrêmes[cite: 3, 4]
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 60)]
    
    return df

def apply_feature_engineering(df, reference_table):
    """Application de la fusion NHANES, encodage et sélection finale[cite: 3, 6]."""
    # 1. Préparation des clés pour la fusion statistique[cite: 2]
    age_bins = [0, 30, 35, 40, 45, 50, 55, 60, 65, 100]
    age_labels = ['0-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65+']
    df['age_group'] = pd.cut(df['age_years'], bins=age_bins, labels=age_labels)
    df['bmi_cat'] = df['bmi'].apply(get_bmi_cat)
    
    # 2. Fusion avec les moyennes nutritionnelles de NHANES[cite: 2, 3]
    df = pd.merge(df, reference_table, on=['age_group', 'bmi_cat'], how='left')
    
    # 3. Encodage One-Hot (Cholesterol et Gluc) avec drop_first pour la multicollinéarité[cite: 3, 6]
    df = pd.get_dummies(df, columns=['cholesterol', 'gluc'], drop_first=True)
    
    # 4. Suppression des variables redondantes (multicollinéarité)[cite: 3, 4, 6]
    # 'weight' et 'height' sont remplacés par 'bmi'
    cols_to_drop = ['id', 'age', 'height', 'weight', 'age_group', 'bmi_cat']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')
    
    return df