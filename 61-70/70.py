# ✅ Ejercicio 70/200 — Clasificación de fake news con arquitectura híbrida CNN + LSTM
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import (
    Input,
    TextVectorization,
    Embedding,
    Conv1D,
    LSTM,
    Dropout,
    Dense,
)
from keras.optimizers import Adam

# -----------------------------
# 📥 1. Carga y preparación de datos
# -----------------------------

# Leer datasets y asignar etiquetas: 0 = fake, 1 = real
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

# Concatenar y limpiar nulos
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

# División estratificada (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# -----------------------------
# 🧹 2. Vectorización del texto
# -----------------------------

# Convierte texto a secuencias de enteros (tokens)
vectorizer = TextVectorization(
    max_tokens=10000,  # Límite de vocabulario
    output_sequence_length=300,  # Padding/recorte a longitud fija
    output_mode="int",
)
vectorizer.adapt(X_train)

X_train_seq = vectorizer(X_train)
X_test_seq = vectorizer(X_test)

# -----------------------------
# 🧠 3. Modelo híbrido CNN + LSTM
# -----------------------------

model = Sequential(
    [
        Input(shape=(300,)),  # Entrada: secuencia de 300 tokens
        Embedding(
            input_dim=10000, output_dim=128
        ),  # Embedding denso aprendido desde cero
        Conv1D(
            filters=128, kernel_size=5, activation="relu"
        ),  # Captura n-gramas locales
        LSTM(64),  # Extrae relaciones secuenciales
        Dropout(0.5),  # Regularización
        Dense(64, activation="relu"),  # Capa oculta densa
        Dropout(0.3),  # Más regularización
        Dense(1, activation="sigmoid"),  # Salida binaria (probabilidad de fake)
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# 🏋️ 4. Entrenamiento
# -----------------------------

model.fit(
    X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
)

# -----------------------------
# 📊 5. Evaluación del modelo
# -----------------------------

y_pred = model.predict(X_test_seq).flatten()
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n📈 Reporte de clasificación CNN + LSTM:\n")
print(classification_report(y_test, y_pred_labels, zero_division=0))
