# ✅ Carga de librerías principales
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (
    TextVectorization,
    Embedding,
    Bidirectional,
    GRU,
    Dense,
    Dropout,
    Input,
)

# 🧾 1. Cargar y preparar el dataset
fake = pd.read_csv("Datasets/archive/Fake.csv")
true = pd.read_csv("Datasets/archive/True.csv")

# Añadir etiquetas manuales: 0 = Fake, 1 = Real
fake["label"] = 0
true["label"] = 1

# Unir los datasets y filtrar columnas relevantes
df = pd.concat([fake, true], ignore_index=True)[["text", "label"]].dropna()

# Extraer características y etiquetas como arrays de numpy
X = df["text"].values
y = df["label"].values

# 🔀 2. Dividir los datos (80% entrenamiento, 20% prueba) con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🔡 3. Vectorización del texto usando Keras TextVectorization
vectorizador = TextVectorization(
    max_tokens=10000,  # Limita el vocabulario a las 10,000 palabras más frecuentes
    output_mode="int",  # Convierte el texto en secuencias de enteros (índices de tokens)
    output_sequence_length=300,  # Longitud fija para todas las secuencias (con padding si es necesario)
)
vectorizador.adapt(
    X_train
)  # Ajusta el vectorizador al vocabulario del conjunto de entrenamiento

# Convertimos el texto en secuencias numéricas
X_train_seq = vectorizador(X_train)
X_test_seq = vectorizador(X_test)

# 🧠 4. Definición del modelo secuencial con Embedding y GRU bidireccional
model = Sequential(
    [
        Input(shape=(300,)),  # Secuencias de 300 tokens por muestra
        Embedding(
            input_dim=10000, output_dim=128
        ),  # Capa de embedding entrenada desde cero
        Bidirectional(
            GRU(64)
        ),  # GRU en ambas direcciones para captar contexto completo
        Dropout(0.3),  # Prevención de sobreajuste
        Dense(
            1, activation="sigmoid"
        ),  # Capa de salida binaria (probabilidad de clase 1)
    ]
)

# ⚙️ 5. Compilar el modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 📈 6. Entrenar el modelo
model.fit(
    X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
)

# 📊 7. Evaluación del modelo
y_pred = model.predict(X_test_seq).flatten()  # Predicciones como probabilidades
y_pred_labels = (y_pred > 0.5).astype(int)  # Clasificación binaria a partir de 0.5

# Reporte de métricas finales
print("\n📋 Reporte de Clasificación — Bidirectional GRU:")
print(classification_report(y_test, y_pred_labels, zero_division=0))
