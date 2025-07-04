# 🧠 Ejercicio 32: Detección de anomalías en Iris usando One-Class SVM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

# 1️⃣ Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target
nombres_clases = iris.target_names

# 2️⃣ Escalar los datos (importante para modelos como SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Entrenar el modelo One-Class SVM
#    - kernel='rbf' para detectar fronteras no lineales
#    - gamma controla la forma de la frontera
#    - nu es la proporción de datos que se espera sean anomalías
ocsvm = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.05)
ocsvm.fit(X_scaled)

# 4️⃣ Predecir etiquetas: 1 (normal), -1 (anómalo)
y_pred = ocsvm.predict(X_scaled)

# 5️⃣ Reducir a 2 dimensiones para visualizar con PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 6️⃣ Localizar las observaciones clasificadas como anomalías
indices_anomalias = np.where(y_pred == -1)

# 7️⃣ Visualizar los resultados en el espacio PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=[nombres_clases[i] for i in y],  # Etiquetas reales
    palette="Set1",
    alpha=0.7,
    s=60,
    edgecolor="k",
    legend="brief",
)

# Agregar los puntos considerados anómalos por el modelo
plt.scatter(
    X_pca[indices_anomalias, 0],
    X_pca[indices_anomalias, 1],
    c="black",
    marker="x",
    s=100,
    label="Anomalías detectadas",
)

plt.title("Detección de Anomalías en Iris con One-Class SVM + PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Clase / Anomalía")
plt.grid(True)
plt.tight_layout()
plt.show()
