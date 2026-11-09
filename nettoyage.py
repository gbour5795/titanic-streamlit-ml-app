import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Les données brutes
data = {
    'client_id': [101, 102, 103, 104, 105, 101],
    'age': [25, np.nan, 29, 150, 42, 25],
    'salaire_k': ['45', '50', 'NaN', '60', '75', '45'],
    'ville': ['Paris', 'Lyon', 'paris', 'Marseille', 'Lyon', 'Paris']
}
df = pd.DataFrame(data)

# --- PHASE 1 : NETTOYAGE (DATA WRANGLING) ---
# On transforme le salaire en vrais nombres
df['salaire_k'] = pd.to_numeric(df['salaire_k'], errors='coerce')
# On supprime l'âge aberrant
df.loc[df['age'] > 100, 'age'] = np.nan
# On uniformise les villes et on supprime les doublons
df['ville'] = df['ville'].str.capitalize()
df = df.drop_duplicates()

# --- PHASE 2 : PRÉPARATION (FEATURE ENGINEERING) ---
# One-Hot Encoding pour transformer le texte des villes en 0 et 1
df = pd.get_dummies(df, columns=['ville'], dtype=int)

# Imputation pour boucher les trous (NaN) dans l'âge et le salaire
imputer = SimpleImputer(strategy='median')
df[['age', 'salaire_k']] = imputer.fit_transform(df[['age', 'salaire_k']])

# --- PHASE 3 : MACHINE LEARNING ---
# Séparation X (indices) et y (cible)
X = df.drop(columns=['salaire_k', 'client_id']) 
y = df['salaire_k'] 

# Découpage train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# L'ÉTAPE MANQUANTE ÉTAIT ICI : Initialisation et entraînement du modèle
modele=LinearRegression()
modele.fit(X_train, y_train)

# Prédiction et Évaluation
y_pred = modele.predict(X_test)
erreur_mae = mean_absolute_error(y_test, y_pred)
score_r2 = r2_score(y_test, y_pred)

print(f"Erreur moyenne de prédiction : {erreur_mae} k€")
print(f"Score R2 (Performance globale) : {score_r2}")