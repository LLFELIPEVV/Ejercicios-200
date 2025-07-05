# 🧠 Ejercicio 35: Detección de anomalías en el dataset Iris usando Autoencoder profundo (Keras)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

# 1️⃣ Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target
nombres_clases = iris.target_names

# 2️⃣ Estandarizar los datos (media 0, varianza 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Definir dimensiones para la arquitectura del autoencoder
input_dim = X_scaled.shape[1]  # 4 variables del iris
encoding_dim = 2  # dimensión del espacio comprimido (cuello de botella)

# 4️⃣ Definir arquitectura del autoencoder profundo
input_layer = Input(shape=(input_dim,), name="Input")

# Codificador: reduce dimensionalidad progresivamente
encoded = Dense(8, activation="relu", name="Encoder_1")(input_layer)
encoded = Dense(4, activation="relu", name="Encoder_2")(encoded)
bottleneck = Dense(encoding_dim, activation="relu", name="Bottleneck")(encoded)

# Decodificador: reconstruye los datos desde la codificación
decoded = Dense(4, activation="relu", name="Decoder_1")(bottleneck)
decoded = Dense(8, activation="relu", name="Decoder_2")(decoded)
output_layer = Dense(input_dim, activation="linear", name="Output")(decoded)

# 5️⃣ Compilar y entrenar el autoencoder
autoencoder = Model(inputs=input_layer, outputs=output_layer, name="Autoencoder")
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss="mse")

autoencoder.fit(
    X_scaled,
    X_scaled,
    epochs=150,
    batch_size=16,
    verbose=0,  # silenciar salida
)

# 6️⃣ Reconstrucción y cálculo del error cuadrático medio por muestra
X_reconstructed = autoencoder.predict(X_scaled)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

# 7️⃣ Determinar umbral para definir anomalías (percentil 95 del error)
threshold = np.percentile(reconstruction_error, 95)
anomalies = reconstruction_error > threshold

# 8️⃣ Visualizar resultados en 2D con PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 9️⃣ Visualización: Clases + Anomalías detectadas
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=[nombres_clases[i] for i in y],
    palette="Set1",
    alpha=0.7,
    s=60,
    edgecolor="k",
    legend="brief",
)

# Dibujar anomalías como cruces negras
plt.scatter(
    X_pca[anomalies, 0],
    X_pca[anomalies, 1],
    color="black",
    marker="x",
    s=100,
    label="Anomalías detectadas",
)

plt.title("Anomalías en Iris con Autoencoder profundo + PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Clase / Anomalía")
plt.grid(True)
plt.tight_layout()
plt.show()
