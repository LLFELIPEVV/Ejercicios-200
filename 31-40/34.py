# 🧠 Ejercicio 34: Detección de Anomalías en Iris usando Autoencoder con Keras

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

# 🔧 Configuración para evitar advertencias de rendimiento en algunas máquinas
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 📥 Cargar y escalar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔍 Definición del Autoencoder: estructura simple con codificador y decodificador
input_dim = X_scaled.shape[1]  # Número de características (4)
encoding_dim = 2  # Compresión en 2 dimensiones

# 🧠 Construcción del modelo Autoencoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)

# 🛠 Compilar y entrenar el modelo
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=16, verbose=0)

# 📤 Obtener las reconstrucciones y calcular el error por instancia
X_reconstructed = autoencoder.predict(X_scaled)
reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

# 🧪 Umbral para considerar una instancia como anómala (percentil 95)
threshold = np.percentile(reconstruction_errors, 95)
anomalies_mask = reconstruction_errors > threshold

# 📊 Reducción de dimensiones para visualización con PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 🎨 Visualización de las clases y las anomalías detectadas
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=[target_names[i] for i in y],
    palette="Set1",
    alpha=0.7,
    s=60,
    edgecolor="k",
)

# 🔴 Puntos detectados como anomalías
plt.scatter(
    X_pca[anomalies_mask, 0],
    X_pca[anomalies_mask, 1],
    color="black",
    marker="x",
    s=100,
    label="Anomalías",
)

# 🖼 Configuración final del gráfico
plt.title("Detección de Anomalías con Autoencoder + PCA (Iris)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Clases / Anomalías")
plt.grid(True)
plt.tight_layout()
plt.show()
