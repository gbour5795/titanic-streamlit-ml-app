import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# 1. ENTRAÎNEMENT DU MODÈLE EN ARRIÈRE-PLAN
# @st.cache_resource garde le modèle en mémoire pour ne pas le recalculer à chaque clic
@st.cache_resource
def charger_modele():
    df = pd.read_csv('train.csv')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # On isole les 4 variables les plus impactantes pour l'interface
    X = df[['Pclass', 'Age', 'Fare', 'Sex']]
    X = pd.get_dummies(X, columns=['Sex'], dtype=int)
    y = df['Survived']
    
    # L'algorithme XGBoost n'a mathématiquement pas besoin de standardisation (contrairement à la PCA)
    modele = XGBClassifier(random_state=42, max_depth=5, learning_rate=0.1, n_estimators=100)
    modele.fit(X, y)
    return modele

modele_xgb = charger_modele()

# 2. INTERFACE VISUELLE
st.title("🚢 Prédicteur de Survie - Titanic")
st.write("Testez l'algorithme : auriez-vous survécu au naufrage ?")

st.sidebar.header("Paramètres du passager")
classe = st.sidebar.selectbox("Classe du billet", [1, 2, 3])
sexe = st.sidebar.radio("Sexe", ["Homme", "Femme"])
age = st.sidebar.slider("Âge", 0, 100, 25)
prix = st.sidebar.number_input("Prix du billet (£)", min_value=0, max_value=500, value=30)

# 3. TRADUCTION DES SAISIES (Feature Engineering en direct)
sex_female = 1 if sexe == "Femme" else 0
sex_male = 1 if sexe == "Homme" else 0

donnees_saisies = pd.DataFrame({
    'Pclass': [classe],
    'Age': [age],
    'Fare': [prix],
    'Sex_female': [sex_female],
    'Sex_male': [sex_male]
})

# 4. PRÉDICTION ET AFFICHAGE
if st.button("Lancer la prédiction de survie"):
    # predict() sort un 0 ou un 1. predict_proba() sort le pourcentage exact.
    prediction = modele_xgb.predict(donnees_saisies)[0]
    probabilite = modele_xgb.predict_proba(donnees_saisies)[0][1] 
    
    if prediction == 1:
        st.success(f"Félicitations, vous survivez avec {probabilite * 100:.1f} % de chances ! 🛶")
    else:
        st.error(f"Désolé, vous coulez. Chances de survie : {probabilite * 100:.1f} % ⚓")