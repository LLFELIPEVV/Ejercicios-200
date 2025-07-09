# ✅ Ejercicio 59/200: Clasificación de noticias reales vs falsas con GloVe + red neuronal en Keras
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

# 📥 Paso 1: Cargar y unir los datasets
df_fake = pd.read_csv(Path("Datasets/archive/Fake.csv"))
df_fake["label"] = "FAKE"

df_true = pd.read_csv(Path("Datasets/archive/True.csv"))
df_true["label"] = "REAL"

df = pd.concat([df_fake, df_true], ignore_index=True)
df = df[["title", "text", "label"]].dropna()  # Solo usamos título y texto

# 🔄 Convertimos etiquetas categóricas a binarias
df["label"] = df["label"].map({"REAL": 1, "FAKE": 0})


# 📚 Paso 2: Cargar GloVe embeddings
def cargar_embeddings(path):
    embeddings = {}
    with open(path, encoding="utf8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embeddings[palabra] = vector
    print(f"✅ Embeddings cargados: {len(embeddings):,}")
    return embeddings


# 🔡 Paso 3: Convertir cada texto en el vector promedio de sus palabras
def texto_a_vector(texto, embeddings, dim=100):
    palabras = texto.lower().split()
    vectores = [embeddings[p] for p in palabras if p in embeddings]
    if not vectores:
        return np.zeros(dim)
    return np.mean(vectores, axis=0)


# 🧠 Paso 4: Preparar datos para el modelo
embedding_path = Path("Gloove/glove.6B.100d.txt")
embedding_index = cargar_embeddings(embedding_path)

X = np.array([texto_a_vector(t, embedding_index) for t in df["text"]])
y = df["label"].values

# (Opcional pero recomendado) Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✂️ Paso 5: Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)


# 🧱 Paso 6: Crear modelo con Keras
def construir_modelo(input_dim):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = construir_modelo(input_dim=100)

# 🏋️ Paso 7: Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# 📊 Paso 8: Evaluación del modelo
y_pred = model.predict(X_test).flatten()
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n📈 Reporte de clasificación:")
print(classification_report(y_test, y_pred_labels, zero_division=0))
