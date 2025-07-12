# ✅ Ejercicio 69/200 — Clasificador de Fake News con CNN 1D sobre embeddings entrenados desde cero
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import (
    Input,
    TextVectorization,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dropout,
    Dense,
)
from keras.optimizers import Adam

# 📥 1. Cargar y etiquetar los datos
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0  # Noticias falsas
df_true["label"] = 1  # Noticias reales

# 🧹 2. Unificación y limpieza
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

# 🔀 3. División estratificada del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🔡 4. Vectorización del texto
vectorizer = TextVectorization(
    max_tokens=10000,  # Límite del vocabulario
    output_mode="int",  # Codifica como enteros
    output_sequence_length=300,  # Secuencias de longitud fija
)
vectorizer.adapt(X_train)  # Aprende el vocabulario del corpus de entrenamiento

# ✏️ 5. Transformar texto a secuencias numéricas
X_train_seq = vectorizer(X_train)
X_test_seq = vectorizer(X_test)

# 🧠 6. Definición del modelo CNN
model = Sequential(
    [
        Input(shape=(300,)),  # Cada input es una secuencia de 300 tokens
        Embedding(input_dim=10000, output_dim=128),  # Embedding entrenable
        Conv1D(
            filters=128, kernel_size=5, activation="relu"
        ),  # Extrae patrones locales
        GlobalMaxPooling1D(),  # Retiene el patrón más fuerte de cada filtro
        Dropout(0.4),  # Previene overfitting
        Dense(64, activation="relu"),  # Capa totalmente conectada
        Dropout(0.3),  # Regularización adicional
        Dense(1, activation="sigmoid"),  # Clasificación binaria
    ]
)

# 🛠️ 7. Compilar el modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 📊 Ver resumen del modelo
model.summary()

# 🚀 8. Entrenamiento
model.fit(
    X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
)

# 📈 9. Evaluación
y_pred = model.predict(X_test_seq).flatten()
y_pred_labels = (y_pred > 0.5).astype(int)

# 📄 10. Reporte de clasificación
print("\nReporte de Clasificación - CNN 1D")
print(classification_report(y_test, y_pred_labels, zero_division=0))
