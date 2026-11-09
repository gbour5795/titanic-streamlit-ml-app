import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Chargement d'un dataset lourd (30 colonnes de mesures médicales)
cancer = load_breast_cancer(as_frame=True)
X = cancer.data

print(f"Format original : {X.shape[1]} variables.")

# 2. Règle d'or absolue de la PCA : TOUJOURS standardiser avant
# Sinon, une variable mesurée en milliers écrasera une variable mesurée en dixièmes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Application de la PCA (Compression de 30 vers 2 dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Vérification de la conservation de l'information (Variance expliquée)
variance_conservee = pca.explained_variance_ratio_

import matplotlib.pyplot as plt

# On récupère la cible (0 = malin, 1 = bénin) pour colorier les points
y = cancer.target

# Création du graphique en nuage de points (Scatter plot)
plt.figure(figsize=(8, 6))
# c=y permet de colorier chaque point selon le diagnostic
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='bwr', alpha=0.7, edgecolors='k')

plt.title("Visualisation 2D des 30 dimensions médicales via PCA")
plt.xlabel(f"Axe Principal 1 ({round(variance_conservee[0]*100, 1)}%)")
plt.ylabel(f"Axe Principal 2 ({round(variance_conservee[1]*100, 1)}%)")

# Ajout d'une légende
plt.legend(handles=scatter.legend_elements()[0], labels=['Malin (Bleu)', 'Bénin (Rouge)'])
plt.show()