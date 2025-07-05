# 🧪 Ejercicio 33: Detección de Anomalías en Iris usando Elliptic Envelope

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA

# 1️⃣ Cargar el dataset Iris
iris = load_iris()
X = iris.data  # Variables numéricas (4)
y = iris.target  # Etiquetas de clase (0, 1, 2)
nombres_clases = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# 2️⃣ Escalado de características para igualar la influencia de todas las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Ajustar el modelo Elliptic Envelope para detección de anomalías
# Contamination indica el % esperado de anomalías en los datos
modelo_ee = EllipticEnvelope(contamination=0.05, random_state=42)
modelo_ee.fit(X_scaled)

# 4️⃣ Predecir las observaciones: 1 = normal, -1 = anomalía
y_pred = modelo_ee.predict(X_scaled)

# 5️⃣ Reducir la dimensionalidad a 2D para visualización
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 6️⃣ Identificar las coordenadas de las observaciones anómalas
indices_anomalias = np.where(y_pred == -1)

# 7️⃣ Visualización de clases reales + anomalías detectadas
plt.figure(figsize=(8, 6))

# Puntos normales coloreados por clase real
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=[nombres_clases[i] for i in y],  # Traducción a etiquetas de clase
    palette="Set1",
    alpha=0.75,
    s=60,
    edgecolor="k",
    legend="brief",
)

# Puntos detectados como anomalías (marcados con X negra)
plt.scatter(
    X_pca[indices_anomalias, 0],
    X_pca[indices_anomalias, 1],
    c="black",
    marker="x",
    s=100,
    label="Anomalías detectadas",
)

# 8️⃣ Configurar la gráfica
plt.title("Detección de Anomalías en Iris con Elliptic Envelope + PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Clase / Anomalía")
plt.grid(True)
plt.tight_layout()
plt.show()
