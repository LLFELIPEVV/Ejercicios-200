# ✅ Ejercicio 66/200 — Clasificación de fake news usando Bidirectional LSTM + Embedding entrenado desde cero
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import (
    TextVectorization,
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dropout,
    Dense,
)
from keras.optimizers import Adam

# ---------------------------
# 1. Cargar y preparar los datos
# ---------------------------

# Cargar noticias falsas y reales
fake = pd.read_csv("Datasets/archive/Fake.csv")
true = pd.read_csv("Datasets/archive/True.csv")

# Asignar etiquetas binarias
fake["label"] = 0  # Noticias falsas
true["label"] = 1  # Noticias reales

# Unir datasets y seleccionar columnas relevantes
df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]].dropna()

# Separar variables de entrada y salida
X = df["text"].values
y = df["label"].values

# Dividir en entrenamiento y prueba de forma estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# 2. Vectorización del texto
# ---------------------------

# Convertir texto en secuencia de enteros (tokens)
vectorizador = TextVectorization(
    max_tokens=10_000,  # Vocabulario máximo
    output_mode="int",  # Salida en índices enteros
    output_sequence_length=300,  # Longitud fija para padding
)
vectorizador.adapt(X_train)  # Aprende el vocabulario desde los datos de entrenamiento

# Aplicar la vectorización al texto
X_train_seq = vectorizador(X_train)
X_test_seq = vectorizador(X_test)

# ---------------------------
# 3. Construcción del modelo con LSTM bidireccional
# ---------------------------

model = Sequential(
    [
        Input(shape=(300,)),  # Secuencia de 300 tokens
        Embedding(input_dim=10_000, output_dim=128),  # Embedding aprendido desde cero
        Bidirectional(LSTM(64)),  # LSTM en ambas direcciones (forward y backward)
        Dropout(0.3),  # Regularización
        Dense(1, activation="sigmoid"),  # Salida binaria
    ]
)

# Compilar el modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Mostrar resumen del modelo
model.summary()

# ---------------------------
# 4. Entrenamiento
# ---------------------------

model.fit(
    X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=2
)

# ---------------------------
# 5. Evaluación
# ---------------------------

# Generar predicciones
y_pred_probs = model.predict(X_test_seq).flatten()
y_pred_labels = (y_pred_probs > 0.5).astype(int)

# Reporte de métricas
print("\n📊 Reporte de clasificación con LSTM Bidireccional:\n")
print(classification_report(y_test, y_pred_labels, zero_division=0))
