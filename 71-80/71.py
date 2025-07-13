# ✅ Ejercicio 71/200 — Comparación práctica: GlobalAveragePooling1D vs GlobalMaxPooling1D
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import (
    Input,
    TextVectorization,
    Embedding,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)
from keras.optimizers import Adam

# -----------------------------
# 📥 1. Carga y preparación de datos
# -----------------------------

# Cargar datasets y asignar etiquetas
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0  # Fake news
df_true["label"] = 1  # Real news

# Unir datasets y eliminar textos vacíos
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()

# Separar variables y etiquetas
X, y = df["text"].values, df["label"].values

# División estratificada: 80% entrenamiento / 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# -----------------------------
# 🧹 2. Vectorización del texto
# -----------------------------

# Capa de procesamiento de texto → secuencia de enteros
vectorizer = TextVectorization(
    max_tokens=10000,  # Tamaño máximo del vocabulario
    output_sequence_length=300,  # Longitud fija de salida (padding o truncado)
    output_mode="int",
)
vectorizer.adapt(X_train)  # Aprende el vocabulario de entrenamiento

# Aplicar vectorización
X_train_seq = vectorizer(X_train)
X_test_seq = vectorizer(X_test)

# -----------------------------
# 🧠 3. Definición de modelos
# -----------------------------


def build_model(pooling="avg"):
    """Construye un modelo con el tipo de pooling especificado."""
    pooling_layer = (
        GlobalAveragePooling1D() if pooling == "avg" else GlobalMaxPooling1D()
    )

    model = Sequential(
        [
            Input(shape=(300,)),
            Embedding(input_dim=10000, output_dim=128),
            pooling_layer,
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


# Crear modelos
model_avg = build_model(pooling="avg")
model_max = build_model(pooling="max")

# -----------------------------
# 🏋️ 4. Entrenamiento de modelos
# -----------------------------

print("🔁 Entrenando modelo con GlobalAveragePooling1D...\n")
model_avg.fit(
    X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
)

print("\n🔁 Entrenando modelo con GlobalMaxPooling1D...\n")
model_max.fit(
    X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
)

# -----------------------------
# 🧪 5. Evaluación y comparación
# -----------------------------


def evaluar_modelo(modelo, nombre):
    """Evalúa un modelo y muestra el classification_report."""
    y_pred = modelo.predict(X_test_seq).flatten()
    y_pred_labels = (y_pred > 0.5).astype(int)

    print(f"\n📊 Reporte de clasificación para {nombre}:\n")
    print(classification_report(y_test, y_pred_labels, zero_division=0))


evaluar_modelo(model_avg, "GlobalAveragePooling1D")
evaluar_modelo(model_max, "GlobalMaxPooling1D")
