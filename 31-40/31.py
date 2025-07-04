# 🧠 Ejercicio 31: Detección de Anomalías en el dataset Iris usando Isolation Forest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# 1️⃣ Cargar el dataset Iris
iris = load_iris()
X = iris.data  # Características (4 variables: largo/ancho sépalo y pétalo)
y = iris.target  # Etiquetas (solo para visualización)

# 2️⃣ Escalar los datos: Isolation Forest se beneficia del escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Entrenar el modelo de aislamiento (Isolation Forest)
# Contamination: fracción esperada de anomalías (5%)
model = IsolationForest(
    n_estimators=100,  # Número de árboles
    contamination=0.05,  # Porcentaje esperado de outliers
    random_state=42,
)
model.fit(X_scaled)

# 4️⃣ Realizar predicciones (-1: anomalía, 1: normal)
y_pred = model.predict(X_scaled)

# 5️⃣ Reducir dimensionalidad con PCA para visualizar en 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 6️⃣ Identificar índices de los puntos anómalos
anomalies_idx = np.where(y_pred == -1)

# 7️⃣ Visualización
plt.figure(figsize=(8, 6))

# Muestra los puntos coloreados por su clase original
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=y,
    palette="Set1",
    alpha=0.7,
    s=60,
    edgecolor="k",
    legend="brief",
)

# Superpone los puntos detectados como anomalías
plt.scatter(
    X_pca[anomalies_idx, 0],
    X_pca[anomalies_idx, 1],
    color="black",
    marker="x",
    s=100,
    label="Anomalías",
)

plt.title("Detección de Anomalías en Iris con Isolation Forest (PCA 2D)")
plt.xlabel("Componente Principal 1 (PC1)")
plt.ylabel("Componente Principal 2 (PC2)")
plt.legend(title="Clase / Anomalía")
plt.grid(True)
plt.tight_layout()
plt.show()
