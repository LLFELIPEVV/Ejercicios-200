# ✅ Ejercicio 65/200 — Clasificación de fake news usando Embedding entrenado + LSTM
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, LSTM, Dropout, Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 1️⃣ Carga de datos reales de noticias
fake = pd.read_csv("Datasets/archive/Fake.csv")
true = pd.read_csv("Datasets/archive/True.csv")

# ✅ Corrección importante: etiquetas correctas
fake["label"] = 0  # Noticia falsa
true["label"] = 1  # Noticia real

# 2️⃣ Unión y limpieza del dataset
df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]].dropna()

X = df["text"].values
y = df["label"].values

# 3️⃣ División estratificada en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4️⃣ Vectorización del texto usando capa TextVectorization
vectorizador = TextVectorization(
    max_tokens=10000,  # Vocabulario máximo
    output_mode="int",  # Salida como índices enteros
    output_sequence_length=300,  # Longitud fija de secuencia
)
vectorizador.adapt(X_train)  # Aprende el vocabulario del corpus

# Transformamos texto a secuencias de enteros
X_train_seq = vectorizador(X_train)
X_test_seq = vectorizador(X_test)

# 5️⃣ Definición del modelo secuencial con LSTM
model = Sequential(
    [
        Input(shape=(300,)),  # Longitud de secuencia esperada
        Embedding(input_dim=10000, output_dim=128),  # Vector entrenable por palabra
        LSTM(64),  # Capa recurrente para captar dependencias contextuales
        Dropout(0.3),  # Regularización
        Dense(1, activation="sigmoid"),  # Clasificación binaria
    ]
)

# 6️⃣ Compilación del modelo
model.compile(
    optimizer=Adam(0.001),  # Optimizador adaptativo
    loss="binary_crossentropy",  # Función de pérdida para clasificación binaria
    metrics=["accuracy"],  # Métrica de evaluación
)

# Mostrar resumen del modelo
model.summary()

# 7️⃣ Entrenamiento con parada temprana
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

model.fit(
    X_train_seq,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
)

# 8️⃣ Evaluación del modelo
y_pred = model.predict(X_test_seq).flatten()
y_pred_labels = (y_pred > 0.5).astype(int)

# 📊 Reporte de desempeño
print(classification_report(y_test, y_pred_labels, zero_division=0))
