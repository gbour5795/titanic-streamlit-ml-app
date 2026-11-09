# 1. VERROUILLAGE DU SYSTÈME (À faire avant tout le reste)
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings('ignore')

# 2. IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 3. CRÉATION DES DONNÉES
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.5, random_state=42)
df = pd.DataFrame(X, columns=['Revenu_Annuel_kE', 'Score_Depense'])

# 4. MÉTHODE DU COUDE
inerties = []
for k in range(1, 11):
    modele_test = KMeans(n_clusters=k, random_state=42, n_init='auto')
    modele_test.fit(df)
    inerties.append(modele_test.inertia_)

# 5. AFFICHAGE DU GRAPHIQUE
plt.plot(range(1, 11), inerties, marker='o')
plt.title('Méthode du coude pour trouver k')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie (Variance intra-cluster)')
plt.show()