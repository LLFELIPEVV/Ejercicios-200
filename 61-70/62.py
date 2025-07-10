# ✅ Ejercicio 62/200 — Entrenamiento de embeddings personalizados desde cero con Keras + TextVectorization
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import (
    TextVectorization,
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Input,
)
from keras.optimizers import Adam

# 🧪 1. Carga de datos reales: noticias verdaderas y falsas
fake = pd.read_csv("Datasets/archive/Fake.csv")
true = pd.read_csv("Datasets/archive/True.csv")

fake["label"] = 0  # Etiqueta 0 → Fake
true["label"] = 1  # Etiqueta 1 → Real

# 🔀 Unimos los datasets y seleccionamos las columnas relevantes
df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]].dropna()

# 🧾 Variables de entrada (X) y salida (y)
X = df["text"].values
y = df["label"].values

# 📚 2. División del dataset en entrenamiento y prueba (estratificado por clase)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ✍️ 3. Vectorización del texto con capa TextVectorization de Keras
vectorizador = TextVectorization(
    max_tokens=10000,  # Máximo número de tokens (palabras distintas)
    output_mode="int",  # Salida como secuencia de enteros
    output_sequence_length=300,  # Secuencias de longitud fija de 300 tokens
)
vectorizador.adapt(X_train)  # Aprende el vocabulario sobre el texto de entrenamiento

# 🔠 Convertimos los textos a secuencias enteras
X_train_seq = vectorizador(X_train)
X_test_seq = vectorizador(X_test)

# 🧠 4. Construcción del modelo secuencial con embedding entrenable desde cero
model = Sequential(
    [
        Input(shape=(300,)),  # Entrada: secuencia de 300 tokens
        Embedding(
            input_dim=10000, output_dim=100
        ),  # Vector de 100 dimensiones por token
        GlobalAveragePooling1D(),  # Promedia los vectores por documento
        Dense(64, activation="relu"),  # Capa densa oculta con ReLU
        Dropout(0.3),  # Dropout para regularización
        Dense(1, activation="sigmoid"),  # Salida binaria (0: fake, 1: real)
    ]
)

# ⚙️ 5. Compilación del modelo: optimizador + función de pérdida + métrica
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 📋 Mostrar arquitectura del modelo
model.summary()

# 🏋️‍♂️ 6. Entrenamiento del modelo
model.fit(
    X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
)

# 🔍 7. Evaluación en datos de prueba
y_pred = model.predict(X_test_seq).flatten()
y_pred_labels = (y_pred > 0.5).astype(int)

# 🧾 8. Reporte de clasificación final
print("\n📊 Reporte de clasificación:\n")
print(classification_report(y_test, y_pred_labels, zero_division=0))
