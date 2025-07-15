# 🧪 Ejercicio 96/200 — Comparación de técnicas de vectorización en modelos densos de Keras

# --------------------------------------------------
# 📦 Importación de librerías necesarias
# --------------------------------------------------
import os
import gc  # Liberar memoria
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam

# --------------------------------------------------
# ⚙️ Configuración del entorno para que TensorFlow use todos los núcleos del procesador
# --------------------------------------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

batch_size = 32  # Fijamos un tamaño de lote bajo para que funcione bien en tu PC

# --------------------------------------------------
# 📥 1. Carga y preparación de los datos
# --------------------------------------------------

# Cargamos dos archivos CSV: noticias falsas y verdaderas
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna()
df_true = pd.read_csv("Datasets/archive/True.csv").dropna()

# Asignamos etiquetas numéricas: 0 = Fake, 1 = Real
df_fake["label"] = 0
df_true["label"] = 1

# Unimos ambos datasets y nos quedamos solo con el texto y la etiqueta
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()

# Separamos características (X) y etiquetas (y)
X, y = df["text"].values, df["label"].values

# Dividimos en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------------------------------
# ✍️ 2. Transformamos el texto en números
# --------------------------------------------------

# 📌 CountVectorizer: cuenta cuántas veces aparece cada palabra (solo frecuencia)
count_vectorizer = CountVectorizer(max_features=5000, stop_words="english")
X_train_count = count_vectorizer.fit_transform(X_train).toarray()
X_test_count = count_vectorizer.transform(X_test).toarray()

# 📌 TfidfVectorizer: mide qué tan importantes son las palabras según su frecuencia en todos los textos
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()


# --------------------------------------------------
# 🧱 3. Definimos el modelo neuronal (igual para ambos casos)
# --------------------------------------------------
def build_model():
    model = Sequential(
        [
            Input(shape=(5000,)),  # Entrada: vector de 5000 dimensiones
            Dense(32, activation="relu"),  # Capa oculta con 32 neuronas y función ReLU
            Dropout(0.1),  # Apagamos el 10% de neuronas aleatoriamente (regularización)
            Dense(
                1, activation="sigmoid"
            ),  # Capa de salida: probabilidad de que sea noticia real (1) o fake (0)
        ]
    )
    model.compile(
        optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


# --------------------------------------------------
# 🤖 4. Entrenamos y evaluamos modelo con CountVectorizer
# --------------------------------------------------
model_count = build_model()
history_count = model_count.fit(
    X_train_count,
    y_train,
    validation_data=(X_test_count, y_test),
    epochs=3,
    batch_size=batch_size,
    verbose=1,
)

# Predicciones: pasamos los valores por el modelo y convertimos la probabilidad a 0 o 1
y_pred_count = (model_count.predict(X_test_count) > 0.5).astype(int)

# Mostramos métricas como precisión, recall, F1, etc.
print("\n📊 Reporte con CountVectorizer:\n")
print(classification_report(y_test, y_pred_count, target_names=["Fake", "Real"]))

# --------------------------------------------------
# 🤖 5. Entrenamos y evaluamos modelo con TfidfVectorizer
# --------------------------------------------------
model_tfidf = build_model()
history_tfidf = model_tfidf.fit(
    X_train_tfidf,
    y_train,
    validation_data=(X_test_tfidf, y_test),
    epochs=3,
    batch_size=batch_size,
    verbose=1,
)

y_pred_tfidf = (model_tfidf.predict(X_test_tfidf) > 0.5).astype(int)

print("\n📊 Reporte con TfidfVectorizer:\n")
print(classification_report(y_test, y_pred_tfidf, target_names=["Fake", "Real"]))

# --------------------------------------------------
# 📊 6. Visualizamos resultados con seaborn
# --------------------------------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))

# Gráfica de la precisión de validación en cada época para ambos modelos
sns.lineplot(
    x=range(1, 4),
    y=history_count.history["val_accuracy"],
    label="CountVectorizer",
    marker="o",
)
sns.lineplot(
    x=range(1, 4),
    y=history_tfidf.history["val_accuracy"],
    label="TfidfVectorizer",
    marker="s",
)

plt.title("Precisión en validación por vectorizador")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.xticks([1, 2, 3])
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 🧹 7. Limpiamos memoria y liberamos recursos
# --------------------------------------------------
del X_train_count, X_test_count, X_train_tfidf, X_test_tfidf
K.clear_session()
gc.collect()
