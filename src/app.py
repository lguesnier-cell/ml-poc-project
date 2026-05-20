from __future__ import annotations
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# Importation de la config pour les chemins officiels

def build_app() -> None:
    st.set_page_config(page_title="CardioCheck ML PoC", layout="wide")

    st.sidebar.title("Navigation Project")
    page = st.sidebar.radio("Aller vers", ["Objectifs & Preprocessing", "Modélisation & Choix", "Démonstration"])

    if page == "Objectifs & Preprocessing":
        show_eda_and_preprocessing()
    elif page == "Modélisation & Choix":
        show_model_logic()
    else:
        show_demo_page()

def show_eda_and_preprocessing():
    st.title("🩺 Objectifs et Pipeline de Preprocessing")
    
    tab_obj, tab_prepro = st.tabs(["Cadrage Business", "Méthode de Preprocessing"])
    
    with tab_obj:
        st.header("1. Problématique Business")
        st.write("""
        L'enjeu est de transformer des données cliniques brutes en un score de risque actionnable. 
        Le but est d'assister le personnel soignant dans le triage des patients pour les maladies cardiovasculaires (MCV).
        """)
        st.info("**Valeur ajoutée :** Fusion des données d'examen classiques avec des indicateurs nutritionnels (NHANES).")

    with tab_prepro:
        st.header("2. Nettoyage et Ingénierie des variables")
        st.write("""
        Le pipeline de préparation suit ces étapes clés :
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Nettoyage (Data Cleaning) :**
            - **Suppression des outliers :** Élimination des valeurs aberrantes de pression artérielle (ex: `ap_hi` > 250 ou < 60) et de poids/taille.
            - **Conversion d'unités :** Transformation de l'âge (jours → années).
            - **Gestion des manquants :** Imputation par la médiane pour les colonnes issues de la fusion NHANES.
            """)
        
        with col2:
            st.markdown("""
            **Feature Engineering :**
            - **Création de l'IMC (BMI) :** Calculé à partir du poids et de la taille.
            - **Catégorisation de la tension :** Création de stades d'hypertension selon les normes médicales.
            - **Encodage :** One-Hot Encoding pour les variables catégorielles (Cholestérol, Glucose).
            """)
    

def show_model_logic():
    st.title("📊 Choix du Modèle et Performances")
    
    st.header("1. Pourquoi ce modèle ?")
    st.write("""
    Nous avons testé trois familles d'algorithmes :
    1. **Régression Logistique :** Baseline simple mais peine sur les relations non-linéaires.
    2. **Random Forest :** Bonne gestion des variables catégorielles, mais moins performant sur le rappel.
    3. **XGBoost (Choisi) :** Puissant algorithme de boosting d'arbres. 
    """)
    
    st.success("""
    **Justification du choix :** XGBoost permet de gérer finement le déséquilibre des classes et offre la meilleure 
    capacité à minimiser les **Faux Négatifs** (patients malades non détectés) via l'optimisation des hyperparamètres.
    """)

    st.header("2. Analyse des Métriques")
    metrics_df = pd.DataFrame([
        {"Modèle": "Baseline (LogReg)",  "Accuracy": 0.7229, "Precision": 0.7478, "Recall": 0.6635, "F1": 0.7031},
        {"Modèle": "Random Forest",      "Accuracy": 0.6849, "Precision": 0.6780, "Recall": 0.6912, "F1": 0.6845},
        {"Modèle": "XGBoost Optimisé",   "Accuracy": 0.5571, "Precision": 0.5280, "Recall": 0.9857, "F1": 0.6876},
    ])
    st.subheader("Comparaison des scores")
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Recall'], color='#90ee90'))
    st.write("**Métrique cible : Rappel (Recall)**. En diagnostic médical, il est préférable d'avoir quelques 'fausses alertes' "
             "plutôt que de passer à côté d'un cas grave.")

    st.header("3. Comparaison Prédiction vs Réalité")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("**Matrice de Confusion**")
        st.image("plot/matrice.png")

    with c2:
        st.write("**Statut Cardiaque par Âge**")
        st.image("plot/statut_cardiaque_age.png")

    with c3:
        st.write("**Variables importantes (XGBoost)**")
        st.image("plot/variables_importantes_XGBoost.png")

def show_demo_page():
    st.title("⚡ Démonstration Interactive")
    
    model_path = Path("models/xgboost_optimized.pkl")
    if not model_path.exists():
        st.error(f"Modèle introuvable à {model_path}")
        return

    model = joblib.load(model_path)
    with st.form("demo_form"):
        st.write("Entrez les données du patient pour une évaluation instantanée :")
        c1, c2, c3 = st.columns(3)
        with c1:
            age    = st.number_input("Âge (ans)", 18, 100, 50)
            weight = st.number_input("Poids (kg)", 30, 200, 75)
            height = st.number_input("Taille (cm)", 130, 220, 175)
            gender = st.selectbox("Sexe", [1, 2], format_func=lambda x: "Femme" if x == 1 else "Homme")
        with c2:
            ap_hi  = st.slider("Tension Systolique (ap_hi)", 80, 220, 120)
            ap_lo  = st.slider("Tension Diastolique (ap_lo)", 50, 120, 80)
            chol   = st.selectbox("Cholestérol", [1, 2, 3], format_func=lambda x: "Normal" if x==1 else "Élevé" if x==2 else "Très élevé")
            gluc   = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: "Normal" if x==1 else "Élevé" if x==2 else "Très élevé")
        with c3:
            smoke  = st.checkbox("Fumeur")
            alco   = st.checkbox("Consommation d'alcool")
            active = st.checkbox("Activité physique régulière")

        submit = st.form_submit_button("Lancer le diagnostic")

    if submit:
        bmi = weight / ((height / 100) ** 2)
        input_df = pd.DataFrame([{
            'gender':        gender,
            'ap_hi':         ap_hi,
            'ap_lo':         ap_lo,
            'smoke':         int(smoke),
            'alco':          int(alco),
            'active':        int(active),
            'age_years':     age,
            'bmi':           bmi,
            'Sodium':        2300.0,
            'Saturated_Fat': 26.0,
            'cholesterol_2': 1 if chol == 2 else 0,
            'cholesterol_3': 1 if chol == 3 else 0,
            'gluc_2':        1 if gluc == 2 else 0,
            'gluc_3':        1 if gluc == 3 else 0,
        }])

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"### Risque détecté : {proba:.1%}")
            st.write("Le patient présente des indicateurs de risque cardiovasculaire élevé.")
        else:
            st.success(f"### Risque faible : {proba:.1%}")
            st.write("Les indicateurs cliniques sont dans la norme.")
if __name__ == "__main__":
    build_app()