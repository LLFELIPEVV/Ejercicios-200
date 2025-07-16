# 🧠 Ejercicio 102/200 — Regularización con L2 y EarlyStopping en un modelo de texto con tf.data.Dataset
import os
import gc
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import TextVectorization, Input, Dense, Dropout

# ======================
# ⚙️ 2. Configurar CPU
# ======================
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(2)
batch_size = 32  # Ideal para Ryzen 3 2200U

# ======================
# 📥 3. Cargar datos
# ======================
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna().sample(500, random_state=42)
df_true = pd.read_csv("Datasets/archive/True.csv").dropna().sample(500, random_state=42)

df_fake["label"] = 0
df_true["label"] = 1
df = pd.concat([df_fake, df_true], ignore_index=True)

X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TextVectorization(max_tokens=5000, output_sequence_length=100)
vectorizer.adapt(tf.convert_to_tensor(X_train))


def prepare_dataset(X, y, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


ds_train = prepare_dataset(vectorizer(tf.convert_to_tensor(X_train)), y_train)
ds_test = prepare_dataset(vectorizer(tf.convert_to_tensor(X_test)), y_test)

model = Sequential(
    [
        Input(shape=(100,)),
        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.1),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(
    ds_train, validation_data=ds_test, epochs=10, callbacks=[early_stop], verbose=1
)

# ==========================
# 📊 8. Evaluación
# ==========================
y_pred = (model.predict(vectorizer(tf.convert_to_tensor(X_test))) > 0.5).astype(int)
print("\n📈 Reporte de clasificación:\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# ==========================
# 📈 9. Gráfico con Seaborn
# ==========================
plt.figure(figsize=(10, 5))
sns.lineplot(
    x=range(1, len(history.history["val_accuracy"]) + 1),
    y=history.history["val_accuracy"],
    label="Validación",
    marker="o",
)
sns.lineplot(
    x=range(1, len(history.history["accuracy"]) + 1),
    y=history.history["accuracy"],
    label="Entrenamiento",
    marker="s",
)
plt.title("Precisión por época con EarlyStopping + L2")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================
# 🧹 10. Limpieza
# ==========================
K.clear_session()
gc.collect()
