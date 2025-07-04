# 🧪 Ejercicio 30: Clustering con Gaussian Mixture Models (GMM) + PCA en el dataset Iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# 1️⃣ Cargar y escalar el conjunto de datos
print("📥 Cargando y normalizando el dataset Iris...")
iris = load_iris()
X, y = iris.data, iris.target
nombres_clases = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2️⃣ Reducción de dimensionalidad para visualización
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("🔍 Varianza explicada por PCA:", pca.explained_variance_ratio_)

# 3️⃣ Evaluar distintos valores de k (componentes GMM) con AIC y BIC
aic_scores = []
bic_scores = []
rango_componentes = range(1, 7)

print("\n🔧 Buscando número óptimo de componentes con AIC/BIC...")
for k in rango_componentes:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm.fit(X_pca)
    bic_scores.append(gmm.bic(X_pca))
    aic_scores.append(gmm.aic(X_pca))

# 4️⃣ Visualizar curva AIC/BIC para elegir número óptimo de componentes
plt.figure(figsize=(6, 4))
plt.plot(rango_componentes, bic_scores, marker="o", label="BIC")
plt.plot(rango_componentes, aic_scores, marker="o", label="AIC")
plt.xlabel("Número de componentes (clusters)")
plt.ylabel("Puntaje AIC / BIC")
plt.title("Selección de número óptimo de clusters con GMM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5️⃣ Entrenar el GMM final con el número óptimo de componentes
k_optimo = np.argmin(bic_scores) + 1  # Seleccionamos el k con menor BIC
print(f"\n✅ Número óptimo de componentes (BIC): {k_optimo}")

gmm_final = GaussianMixture(
    n_components=k_optimo, covariance_type="full", random_state=42
)
gmm_final.fit(X_pca)
y_gmm = gmm_final.predict(X_pca)

# 6️⃣ Evaluar calidad del clustering usando el ARI
ari = adjusted_rand_score(y, y_gmm)
print(f"🎯 ARI (Adjusted Rand Index) comparado con clases reales: {ari:.4f}")

# 7️⃣ Visualización de los clusters con sus elipses gaussianas
plt.figure(figsize=(8, 6))
ax = sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1], hue=y_gmm, palette="Set2", edgecolor="k", alpha=0.8
)

# Dibujar elipses de las gaussianas (1σ y 2σ)
for mean, cov in zip(gmm_final.means_, gmm_final.covariances_):
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    for scale in [1, 2]:  # Dibujar 1σ y 2σ
        width, height = 2 * scale * np.sqrt(eigvals)
        ellipse = plt.matplotlib.patches.Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor="k",
            facecolor="none",
            linestyle="--" if scale == 2 else "-",
        )
        ax.add_patch(ellipse)

plt.title("Clustering con GMM en espacio PCA - Iris")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Cluster asignado por GMM")
plt.grid(True)
plt.tight_layout()
plt.show()
