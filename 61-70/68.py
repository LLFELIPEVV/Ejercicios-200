# ✅ Ejercicio 68/200 — Clasificador de fake news con Bidirectional LSTM + Dropout + BatchNormalization
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (
    Input,
    TextVectorization,
    Embedding,
    Bidirectional,
    LSTM,
    BatchNormalization,
    Dropout,
    Dense,
)

# 📥 Cargar datos
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")

# 🏷️ Asignar etiquetas
df_fake["label"] = 0  # Fake news
df_true["label"] = 1  # Real news

# 🧹 Unir y limpiar datos
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

# 🧪 División estratificada para conservar proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ✏️ Vectorización del texto (tokenización + secuencias fijas)
vectorizer = TextVectorization(
    max_tokens=10000, output_mode="int", output_sequence_length=300
)
vectorizer.adapt(X_train)

X_train_seq = vectorizer(X_train)
X_test_seq = vectorizer(X_test)

# 🧠 Arquitectura del modelo
model = Sequential(
    [
        Input(shape=(300,)),  # Secuencia de 300 enteros
        Embedding(input_dim=10000, output_dim=128),  # Embedding entrenado desde cero
        Bidirectional(LSTM(64, return_sequences=False)),  # LSTM bidireccional
        BatchNormalization(),  # Estabiliza el entrenamiento
        Dropout(0.4),  # Regulariza activaciones del BiLSTM
        Dense(64, activation="relu"),  # Capa intermedia
        Dropout(0.3),  # Regulariza la capa densa
        Dense(1, activation="sigmoid"),  # Salida binaria
    ]
)

# ⚙️ Compilación del modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 📊 Entrenamiento con validación
model.fit(X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 📈 Predicción y evaluación
y_pred = model.predict(X_test_seq).flatten()
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n📋 Reporte de clasificación (Fake vs Real):\n")
print(classification_report(y_test, y_pred_labels, zero_division=0))
