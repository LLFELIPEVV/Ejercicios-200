# 🧪 Ejercicio 82/200 — Clasificación de Fake News con Keras
# ------------------------------------------------------------
# Objetivo: Entrenar una red neuronal densa para detectar noticias falsas,
# usando vectores TF-IDF como representación del texto.
# ------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# ------------------------------------------------------------
# 📥 1. Cargar y preparar el dataset
# ------------------------------------------------------------

# Carga de los archivos
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")

# Etiquetar: 0 = Fake, 1 = Real
df_fake["label"] = 0
df_true["label"] = 1

# Unir ambos datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# Mantener solo columnas necesarias y eliminar nulos
df = df[["text", "label"]].dropna()

# Separar variables
X = df["text"].values
y = df["label"].values

# ------------------------------------------------------------
# ✂️ 2. División de los datos (entrenamiento y test)
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # Mantiene proporción de clases
)

# ------------------------------------------------------------
# 🔠 3. Vectorización del texto (TF-IDF)
# ------------------------------------------------------------

# Configurar el vectorizador
vectorizer = TfidfVectorizer(
    stop_words="english",  # Elimina palabras poco informativas
    max_features=10000,  # Limita el vocabulario para eficiencia
)

# Entrenar el vectorizador solo con el set de entrenamiento
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()

# Aplicar transformación al set de test (sin reentrenar)
X_test_tfidf = vectorizer.transform(X_test).toarray()

# ------------------------------------------------------------
# 🧠 4. Definición del modelo en Keras
# ------------------------------------------------------------

# Arquitectura de red neuronal feedforward
model = Sequential(
    [
        Dense(
            256, activation="relu", input_shape=(X_train_tfidf.shape[1],)
        ),  # Capa densa con ReLU
        Dropout(0.4),  # Prevención de sobreajuste
        Dense(64, activation="relu"),  # Capa intermedia más pequeña
        Dropout(0.3),  # Otro nivel de regularización
        Dense(1, activation="sigmoid"),  # Capa de salida binaria
    ]
)

# Compilar el modelo
model.compile(
    optimizer=Adam(learning_rate=1e-3),  # Optimizador Adam con LR explícito
    loss="binary_crossentropy",  # Pérdida para clasificación binaria
    metrics=["accuracy"],  # Métrica de desempeño
)

# ------------------------------------------------------------
# 🏋️ 5. Entrenamiento del modelo
# ------------------------------------------------------------

# Early stopping para evitar sobreajuste si no mejora la validación
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    X_train_tfidf,
    y_train,
    validation_split=0.1,  # 10% del train se usa como validación
    epochs=15,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2,  # Muestra entrenamiento más legible
)

# ------------------------------------------------------------
# 📊 6. Evaluación del modelo
# ------------------------------------------------------------

# Predicciones de probabilidad
y_pred_probs = model.predict(X_test_tfidf, verbose=0)

# Conversión a clase (umbral 0.5)
y_pred = (y_pred_probs >= 0.5).astype(int)

# 📈 Reporte de clasificación
print("\n📈 Reporte de clasificación (Keras):")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# 🧾 Matriz de confusión
print("🧾 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
