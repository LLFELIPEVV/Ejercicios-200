# 🧪 Ejercicio 97/200 — Comparación: TextVectorization (Keras) vs TfidfVectorizer (Scikit-learn)

# 🔧 Importamos las librerías necesarias
import os
import gc  # para liberar memoria
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import backend as K
from keras.models import Sequential
from keras.layers import TextVectorization, Dense, Input, Dropout
from keras.optimizers import Adam

# 🧠 Configuramos TensorFlow para usar todos los núcleos del procesador (Ryzen 3 2200U)
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

# 📦 Límite razonable de procesamiento
batch_size = (
    32  # Tamaño de grupo de datos que se procesan en cada paso del entrenamiento
)

# 📄 Cargamos las noticias falsas y reales desde archivos CSV
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna()
df_true = pd.read_csv("Datasets/archive/True.csv").dropna()

# 🏷️ Asignamos etiquetas: 0 = Fake, 1 = Real
df_fake["label"] = 0
df_true["label"] = 1

# 🔄 Combinamos ambos conjuntos y tomamos una muestra de 1000 para optimizar rendimiento
df = pd.concat([df_fake, df_true])[["text", "label"]].sample(1000, random_state=42)

# 🔢 Separamos los textos y sus etiquetas
X = df["text"].values  # Textos
y = df["label"].values  # Etiquetas

# ✂️ Dividimos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# -----------------------
# 🔤 Vectorización con Scikit-learn (TF-IDF)
# -----------------------
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

# -----------------------
# 🔤 Vectorización con Keras (TextVectorization)
# -----------------------

# Convertimos los textos a Tensores para usarlos con Keras
X_train_tensor = tf.convert_to_tensor(X_train)
X_test_tensor = tf.convert_to_tensor(X_test)

# Creamos la capa que transformará texto en vectores TF-IDF
text_vectorizer = TextVectorization(
    max_tokens=5000,  # Número máximo de palabras
    output_mode="tf_idf",  # Salida similar al TfidfVectorizer
)

# "Adaptamos" el vectorizador: aprende las palabras del conjunto de entrenamiento
text_vectorizer.adapt(X_train_tensor)

# Transformamos los textos en vectores usando la capa entrenada
X_train_keras = text_vectorizer(X_train_tensor).numpy()
X_test_keras = text_vectorizer(X_test_tensor).numpy()


# -----------------------
# 🧠 Definimos el modelo neuronal
# -----------------------
def build_model(input_dim):
    model = Sequential(
        [
            Input(shape=(input_dim,)),  # Entrada del modelo: vector de tamaño 5000
            Dense(32, activation="relu"),  # Capa oculta con 32 neuronas y función ReLU
            Dropout(
                0.1
            ),  # Apaga aleatoriamente el 10% de las neuronas para evitar sobreajuste
            Dense(1, activation="sigmoid"),  # Salida binaria: predice 0 o 1
        ]
    )
    model.compile(
        optimizer=Adam(1e-3),
        loss="binary_crossentropy",  # Usamos esta función para clasificación binaria
        metrics=["accuracy"],
    )
    return model


# -----------------------
# 🚀 Entrenamiento con TfidfVectorizer
# -----------------------
model_sklearn = build_model(5000)
history_sklearn = model_sklearn.fit(
    X_train_tfidf,
    y_train,
    validation_data=(X_test_tfidf, y_test),
    epochs=3,
    batch_size=batch_size,
    verbose=1,
)

# 🔍 Predicciones y evaluación del modelo
y_pred_sklearn = (model_sklearn.predict(X_test_tfidf) > 0.5).astype(int)
print("\nReporte con TfidfVectorizer (Scikit-learn):\n")
print(classification_report(y_test, y_pred_sklearn, target_names=["Fake", "Real"]))

# -----------------------
# 🚀 Entrenamiento con TextVectorization
# -----------------------
model_keras = build_model(5000)
history_keras = model_keras.fit(
    X_train_keras,
    y_train,
    validation_data=(X_test_keras, y_test),
    epochs=3,
    batch_size=batch_size,
    verbose=1,
)

# 🔍 Predicciones y evaluación
y_pred_keras = (model_keras.predict(X_test_keras) > 0.5).astype(int)
print("\nReporte con TextVectorization (Keras):\n")
print(classification_report(y_test, y_pred_keras, target_names=["Fake", "Real"]))

# -----------------------
# 📈 Visualización de los resultados
# -----------------------
plt.figure(figsize=(10, 6))

# Seaborn con nombre de épocas en eje X
sns.lineplot(
    x=range(1, 4),  # Épocas 1, 2, 3
    y=history_sklearn.history["val_accuracy"],
    label="TfidfVectorizer (Sklearn)",
    marker="o",
    linewidth=2,
)

sns.lineplot(
    x=range(1, 4),
    y=history_keras.history["val_accuracy"],
    label="TextVectorization (Keras)",
    marker="s",
    linewidth=2,
)

plt.title("Precisión de validación por época", fontsize=14)
plt.xlabel("Época", fontsize=12)
plt.ylabel("Precisión", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks([1, 2, 3])
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------
# 🧹 Limpiamos la sesión para liberar memoria
# -----------------------
K.clear_session()
gc.collect()
