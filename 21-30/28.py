# 🧪 Ejercicio 28: Análisis PCA + Agrupamiento con KMeans en el conjunto Iris
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# 1️⃣ Cargar el conjunto de datos Iris (flores con 4 características)
iris = load_iris()
X = iris.data  # Variables predictoras (longitudes y anchos)
y = iris.target  # Etiquetas reales (clases de flor)
nombres_clases = iris.target_names

# 2️⃣ Normalización estándar (media = 0, desviación estándar = 1)
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# 3️⃣ Reducción de dimensionalidad con PCA a 2 componentes principales
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_normalizado)

print("📊 Varianza explicada por cada componente PCA:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"   Componente {i + 1}: {ratio:.4f}")

# 4️⃣ Aplicar KMeans para encontrar 3 clusters (esperamos que coincidan con las 3 clases reales)
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
etiquetas_cluster = kmeans.fit_predict(X_pca)

# 5️⃣ Evaluar qué tanto coinciden los clusters con las clases verdaderas usando ARI
ari = adjusted_rand_score(y, etiquetas_cluster)
print(f"\n🔍 Adjusted Rand Index (clusters vs clases reales): {ari:.4f}")

# 6️⃣ Visualización comparativa: clases reales vs clusters encontrados
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 🔹 Visualizar clases reales
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="Set1", edgecolor="k", alpha=0.8)
axes[0].set_title("Especies reales - Iris")
axes[0].set_xlabel("Componente Principal 1")
axes[0].set_ylabel("Componente Principal 2")
axes[0].grid(True)

# 🔹 Visualizar clusters encontrados por KMeans
axes[1].scatter(
    X_pca[:, 0], X_pca[:, 1], c=etiquetas_cluster, cmap="Set1", edgecolor="k", alpha=0.8
)
axes[1].scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="white",
    s=250,
    marker="X",
    edgecolor="black",
    linewidth=1.5,
    label="Centroides",
)
axes[1].set_title("Clusters encontrados por KMeans")
axes[1].set_xlabel("Componente Principal 1")
axes[1].set_ylabel("Componente Principal 2")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("PCA + KMeans sobre el conjunto Iris")
plt.tight_layout()
plt.show()
