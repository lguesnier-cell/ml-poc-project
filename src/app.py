from __future__ import annotations
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# Importation de la config pour les chemins officiels
from config import MODEL_METRICS_FILE

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
        D'après nos analyses dans le notebook, le pipeline de préparation suit ces étapes clés :
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
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Data_preprocessing_steps.png/600px-Data_preprocessing_steps.png", width=400, caption="Schéma du flux de données")

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
    if MODEL_METRICS_FILE.exists():
        metrics_df = pd.read_csv(MODEL_METRICS_FILE)
        st.subheader("Comparaison des scores")
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['recall'], color='#90ee90'))
        
        st.write("**Métrique cible : Rappel (Recall)**. En diagnostic médical, il est préférable d'avoir quelques 'fausses alertes' "
                 "plutôt que de passer à côté d'un cas grave.")
    else:
        st.warning("Fichier de métriques introuvable. Avez-vous lancé l'entraînement ?")

    st.header("3. Comparaison Prédiction vs Réalité")
    res_path = Path("results/test_results.csv")
    if res_path.exists():
        res_df = pd.read_csv(res_path)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Matrice de Confusion**")
            cm = confusion_matrix(res_df['y_true'], res_df['y_pred'])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
            st.pyplot(fig)
        with c2:
            st.write("**Importance des Variables (Feature Importance)**")
            st.write("Le modèle accorde le plus de poids à : **Pression Systolique**, **Âge** et **Cholestérol**.")
    else:
        st.info("Ajoutez un fichier 'results/test_results.csv' pour visualiser la performance réelle.")

def show_demo_page():
    st.title("⚡ Démonstration Interactive")
    
    model_path = Path("models/xgboost_optimized.pkl")
    if not model_path.exists():
        st.error(f"Modèle introuvable à {model_path}")
        return

    model = joblib.load(model_path)

    with st.form("demo_form"):
        st.write("Entrez les données du patient pour une évaluation instantanée :")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Âge (ans)", 18, 100, 50)
            weight = st.number_input("Poids (kg)", 30, 200, 75)
            height = st.number_input("Taille (cm)", 130, 220, 175)
        with c2:
            ap_hi = st.slider("Tension Systolique (ap_hi)", 80, 220, 120)
            chol = st.selectbox("Cholestérol", [1, 2, 3], format_func=lambda x: "Normal" if x==1 else "Élevé" if x==2 else "Très élevé")
            smoke = st.checkbox("Fumeur")

        submit = st.form_submit_button("Lancer le diagnostic")

    if submit:
        # Reprise du preprocessing simplifié pour la démo
        bmi = weight / ((height/100)**2)
        # Création du vecteur d'entrée compatible avec le modèle
        input_df = pd.DataFrame([{
            'age_years': age,
            'ap_hi': ap_hi,
            'bmi': bmi,
            'smoke': int(smoke),
            'cholesterol_2': 1 if chol == 2 else 0,
            'cholesterol_3': 1 if chol == 3 else 0
        }])
        
        # Complétion des colonnes manquantes si nécessaire (XGBoost en a besoin)
        # Note: ceci est un exemple, ajustez selon les colonnes exactes de votre modèle
        
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