import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Chargement
df = pd.read_csv('train.csv')

# 2. Data Wrangling (Nettoyage)
df = df.drop(columns=['Cabin'])
df = df.dropna(subset=['Embarked'])
imputer = SimpleImputer(strategy='median')
df[['Age']] = imputer.fit_transform(df[['Age']])

# 3. Feature Engineering (Encodage et purge)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int)
df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])

# 4. Définition X et y
X = df.drop(columns=['Survived'])
y = df['Survived']

# 5. Split (Séparation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Standardisation (avec l'astuce pour garder les noms de colonnes)
scaler = StandardScaler()
scaler.set_output(transform="pandas")
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # Attention: toujours seulement .transform() sur le test

# 7. Le Modèle (La Forêt Aléatoire bridée pour éviter le surapprentissage)
modele = XGBClassifier(random_state=42, max_depth=5, learning_rate=0.1, n_estimators=100)

# 8. Évaluation robuste : La Validation Croisée (Crash-test en 5 examens)
scores_cv = cross_val_score(modele, X, y, cv=5, scoring='accuracy')

# 1. On entraîne le modèle une bonne fois pour toutes sur nos données d'entraînement
modele.fit(X_train, y_train)

# 2. On extrait l'importance mathématique (en pourcentages)
importances = modele.feature_importances_

# 3. On crée un petit tableau propre pour afficher le podium
df_importances = pd.DataFrame({
    'Variable': X.columns,
    'Importance': importances * 100 # Multiplié par 100 pour lire en pourcentages
}).sort_values(by='Importance', ascending=False)

scores_cv = cross_val_score(modele, X, y, cv=5, scoring='accuracy')

print("Détail des 5 examens XGBoost :", scores_cv)
print(f"Score moyen certifié XGBoost : {scores_cv.mean() * 100:.2f} %")