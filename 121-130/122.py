# 🧪 Ejercicio 122/200 — Visualización de desbalance de clases y preparación eficiente del conjunto de entrenamiento
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Creamos etiquetas simuladas: 80% reales (0), 20% falsas (1)
labels = np.concatenate(
    [
        np.zeros(800, dtype=int),  # Clase 0: noticias reales
        np.ones(200, dtype=int),  # Clase 1: noticias falsas
    ]
)

# 2️⃣ Visualizamos el desbalance usando Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.countplot(x=labels, palette=["#5DADE2", "#E74C3C"])
plt.title("Distribución de clases en el dataset (0 = real, 1 = fake)")
plt.xlabel("Clase")
plt.ylabel("Cantidad de ejemplos")
plt.show()

# 3️⃣ Simulamos features (entradas) como strings de texto falso
texts = ["Fake news example"] * 200 + ["Real news example"] * 800

# 4️⃣ Convertimos a un tf.data.Dataset con texto y etiquetas
raw_ds = tf.data.Dataset.from_tensor_slices((texts, labels))

# 5️⃣ Aplicamos transformaciones para optimizar el entrenamiento
# Nota: aquí solo simulamos el pipeline, no usamos el modelo aún

BATCH_SIZE = 32

dataset = (
    raw_ds.shuffle(buffer_size=1000)  # Mezcla aleatoriamente los datos
    .batch(BATCH_SIZE)  # Agrupa en lotes de 32
    .cache()  # Guarda datos transformados en RAM
    .prefetch(
        tf.data.AUTOTUNE
    )  # Prepara el siguiente lote mientras se entrena el actual
)

# 6️⃣ Mostramos los primeros 2 lotes para confirmar que se generó correctamente
for i, (text_batch, label_batch) in enumerate(dataset.take(2)):
    print(f"\n🔢 Lote {i + 1}")
    print("📝 Textos:", text_batch.numpy()[:3])  # Muestra solo los primeros 3 textos
    print(
        "🏷️ Etiquetas:", label_batch.numpy()[:3]
    )  # Muestra solo las primeras 3 etiquetas
