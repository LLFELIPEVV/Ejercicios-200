# ✅ Ejercicio 73/200 — Visualización de pesos del Embedding Layer entrenado desde cero para análisis interpretativo
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import (
    Input,
    TextVectorization,
    Embedding,
    GlobalAveragePooling1D,
    Dense,
)
from keras.optimizers import Adam

# -------------------------------
# 📥 1. Carga y preparación de datos
# -------------------------------

# Lectura de archivos
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")

# Etiquetado de clases: 0 = fake, 1 = real
df_fake["label"] = 0
df_true["label"] = 1

# Unión y limpieza
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 🔠 2. Vectorización del texto
# -------------------------------

# Configuración del vectorizador
vectorizer = TextVectorization(
    max_tokens=10000,  # Límite de vocabulario
    output_sequence_length=300,  # Secuencia fija
    output_mode="int",  # Secuencias como enteros
)
vectorizer.adapt(X_train)  # Aprende el vocabulario de entrenamiento

# Transformar texto a secuencias
X_train_seq = vectorizer(X_train)

# -------------------------------
# 🧠 3. Definición y entrenamiento del modelo
# -------------------------------

model = Sequential(
    [
        Input(shape=(300,)),  # Entrada de secuencia fija
        Embedding(
            input_dim=10000, output_dim=100, name="embedding"
        ),  # Capa de embedding entrenada desde cero
        GlobalAveragePooling1D(),  # Reduce a vector fijo por promedio
        Dense(1, activation="sigmoid"),  # Capa de salida binaria
    ]
)

model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Entrenamiento (solo 3 épocas para análisis interpretativo)
model.fit(X_train_seq, y_train, epochs=3, batch_size=32)

# -------------------------------
# 🔍 4. Extracción de pesos del embedding
# -------------------------------

# Extraer pesos aprendidos de la capa de embedding
embedding_weights = model.get_layer("embedding").get_weights()[0]  # Shape: (10000, 100)

# Obtener el vocabulario aprendido por el vectorizador
vocab = vectorizer.get_vocabulary()

# Seleccionar las 300 palabras más frecuentes
num_words = 300
selected_embeddings = embedding_weights[:num_words]
selected_vocab = vocab[:num_words]

# -------------------------------
# 🎯 5. Reducción de dimensionalidad con t-SNE
# -------------------------------

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced = tsne.fit_transform(selected_embeddings)

# -------------------------------
# 📊 6. Visualización de embeddings en 2D
# -------------------------------

plt.figure(figsize=(12, 10))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)

# Etiquetar puntos con las palabras
for i, word in enumerate(selected_vocab):
    plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=8, alpha=0.7)

plt.title("📌 Proyección 2D de Embeddings Entrenados desde Cero")
plt.grid(True)
plt.tight_layout()
plt.show()
