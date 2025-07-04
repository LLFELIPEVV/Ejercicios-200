# 🧪 Agrupamiento con DBSCAN + Reducción de Dimensiones (PCA) sobre el dataset Iris
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

# 1️⃣ Cargar y preparar datos
print("📥 Cargando dataset Iris...")
iris = load_iris()
X, y = iris.data, iris.target
nombres_clases = iris.target_names

# 2️⃣ Normalización (escalado estándar con media 0 y desviación estándar 1)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# 3️⃣ Reducción de dimensiones con PCA a 2 componentes principales
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_norm)
print("\n📊 Varianza explicada por PCA:", pca.explained_variance_ratio_)

# 4️⃣ Agrupamiento con DBSCAN (no requiere número de clusters)
dbscan = DBSCAN(eps=0.5, min_samples=5)
etiquetas_dbscan = dbscan.fit_predict(X_pca)

# 5️⃣ Evaluar calidad del agrupamiento con ARI (solo si hay clusters)
clusters_encontrados = len(set(etiquetas_dbscan)) - (1 if -1 in etiquetas_dbscan else 0)

if clusters_encontrados >= 2:
    ari = adjusted_rand_score(y, etiquetas_dbscan)
    print(f"\n✅ Clusters encontrados (sin ruido): {clusters_encontrados}")
    print(f"🔍 Adjusted Rand Index (ARI): {ari:.4f}")
else:
    print("\n⚠️ No se encontraron suficientes clusters para calcular ARI.")

# 6️⃣ Visualización de los clusters encontrados (en el espacio PCA)
plt.figure(figsize=(8, 6))
colores = plt.cm.get_cmap("tab10", clusters_encontrados + 1)

for etiqueta in set(etiquetas_dbscan):
    mascara = etiquetas_dbscan == etiqueta
    if etiqueta == -1:
        # Clase -1 representa ruido detectado por DBSCAN
        plt.scatter(
            X_pca[mascara, 0],
            X_pca[mascara, 1],
            c="gray",
            label="Ruido",
            edgecolor="k",
            alpha=0.5,
            s=60,
        )
    else:
        plt.scatter(
            X_pca[mascara, 0],
            X_pca[mascara, 1],
            label=f"Cluster {etiqueta}",
            edgecolor="k",
            alpha=0.8,
            s=80,
            c=[colores(etiqueta)],
        )

plt.title("Agrupamiento con DBSCAN - Iris (PCA 2D)", fontsize=14)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
