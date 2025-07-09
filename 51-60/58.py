# ✅ Ejercicio 58/200: Clasificación binaria de noticias reales vs falsas con GloVe + Keras
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


# 📥 Cargar embeddings GloVe desde archivo
def cargar_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    embeddings = {}
    with open(path, encoding="utf8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embeddings[palabra] = vector
    return embeddings


# 🔠 Convertir texto a vector por promedio de embeddings
def texto_a_vector(texto, embeddings, dim=100):
    palabras = texto.lower().split()
    vectores = [embeddings[p] for p in palabras if p in embeddings]
    if not vectores:
        return np.zeros(dim)
    return np.mean(vectores, axis=0)


# 📰 Dataset simulado: noticias reales y fake
noticias = [
    {"texto": "Pfizer anuncia eficacia del 95% en su vacuna", "etiqueta": 1},
    {
        "texto": "Científicos descubren nueva cepa de COVID más contagiosa",
        "etiqueta": 1,
    },
    {"texto": "¡El fin del mundo llega en 2023 según los mayas!", "etiqueta": 0},
    {"texto": "Bill Gates planea controlar el clima con satélites", "etiqueta": 0},
    {"texto": "La NASA encuentra agua en la Luna", "etiqueta": 1},
    {"texto": "¡La Tierra es plana! Lo confirma estudio oculto", "etiqueta": 0},
    {"texto": "Científicos colombianos crean energía con agua sucia", "etiqueta": 1},
    {"texto": "Nuevo chip 6G permitirá leer tus pensamientos", "etiqueta": 0},
]

# 🧠 Preprocesamiento
glove_path = "Gloove/glove.6B.100d.txt"
embedding_index = cargar_embeddings(glove_path)

X = np.array([texto_a_vector(n["texto"], embedding_index) for n in noticias])
y = np.array([n["etiqueta"] for n in noticias])

# 🔀 División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 🧱 Modelo secuencial simple
model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ]
)

# ⚙️ Compilación
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 🏋️ Entrenamiento
model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=2,
    verbose=1,
    validation_data=(X_test, y_test),
)

# 🧪 Evaluación
y_pred = model.predict(X_test).flatten()
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n📊 Reporte de clasificación:\n")
print(classification_report(y_test, y_pred_labels, zero_division=0))
