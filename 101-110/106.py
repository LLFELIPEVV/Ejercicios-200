# 🧠 Ejercicio 106/200 — Eficiencia con tf.data.Dataset: lectura, preparación y comparación

# ===================================
# 📦 1. Importar librerías necesarias
# ===================================
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization

# ==========================================
# 📚 2. Cargar datos y etiquetar (fake vs real)
# ==========================================
# Cargamos 500 noticias falsas y 500 reales
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna().sample(500, random_state=42)
df_true = pd.read_csv("Datasets/archive/True.csv").dropna().sample(500, random_state=42)

# Etiquetas: 0 = Fake, 1 = Real
df_fake["label"] = 0
df_true["label"] = 1

# Unimos en un solo DataFrame con solo texto y etiqueta
df = pd.concat([df_fake, df_true])[["text", "label"]]

# =============================
# ✂️ 3. Separar en train/test
# =============================
X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ===========================================
# ✍️ 4. Vectorización de texto con Keras
# ===========================================
vocab_size = 5000
max_len = 100

vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=max_len)
vectorizer.adapt(tf.convert_to_tensor(X_train))  # Aprende el vocabulario


# =======================================
# ⚙️ 5. Crear función para mapear (texto, etiqueta) → vector
# =======================================
def preprocess(text, label):
    text = tf.expand_dims(
        text, -1
    )  # Requiere expansión para que el vectorizer funcione
    return vectorizer(text), label


# ===============================================
# 🧪 6. Crear datasets usando tf.data.Dataset
# ===============================================
BATCH_SIZE = 32

# Dataset de entrenamiento
ds_train = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=1000)
    .map(preprocess)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Dataset de prueba
ds_test = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .map(preprocess)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# =================================================
# 🔁 7. Medir tiempo usando listas vs tf.data.Dataset
# =================================================
# En listas (convertido todo en memoria)
X_train_vec = vectorizer(tf.convert_to_tensor(X_train))
X_test_vec = vectorizer(tf.convert_to_tensor(X_test))

start_list = time.time()
_ = list(zip(X_train_vec, y_train))
end_list = time.time()

# Con tf.data.Dataset (ya optimizado)
start_ds = time.time()
for batch in ds_train:
    pass
end_ds = time.time()

# Mostrar resultados
sns.barplot(
    x=["Listas (memoria)", "tf.data.Dataset (streaming)"],
    y=[end_list - start_list, end_ds - start_ds],
)
plt.title("Tiempo de preparación de datos")
plt.ylabel("Tiempo (segundos)")
plt.show()
