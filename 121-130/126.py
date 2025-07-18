# ✅ Ejercicio 126/200 — Visualización + Tokenización subword personalizada + Exportación a producción
# Paso 1️⃣: Importar librerías necesarias
import re  # Para limpiar texto con expresiones regulares
import numpy as np  # Para cálculos matemáticos eficientes
import pandas as pd  # Para manejar datos en forma de tabla
import tensorflow as tf  # Framework principal para Deep Learning
import seaborn as sns  # Para gráficos más bonitos y claros
import matplotlib.pyplot as plt  # Para mostrar gráficos

# Componentes de Keras para construir la red neuronal
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    TextVectorization,
)
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)  # Callbacks para evitar sobreentrenamiento

# Herramientas de Scikit-learn para partición y balanceo de clases
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Paso 2️⃣: Crear un pequeño dataset simulado con noticias reales y falsas
data = {
    "text": [
        "Experts confirm the truth",
        "Aliens land in garden!",
        "Official source says yes",
        "This is a hoax",
        "Tea proven to help health",
        "President fakes numbers",
        "Confirmed by data",
        "Scam alert: FAKE!",
        "Bananas cure diseases!",
        "WHO confirms science",
    ],
    "label": [
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "fake",
        "real",
    ],
}
df = pd.DataFrame(data)


# Paso 3️⃣: Función de limpieza de texto usando expresiones regulares
def limpiar(texto):
    texto = texto.lower()  # Convertir todo a minúsculas
    texto = re.sub(r"http\S+", "", texto)  # Quitar URLs
    texto = re.sub(r"\d+", "", texto)  # Quitar números
    texto = re.sub(r"[^a-z\s]", "", texto)  # Quitar signos de puntuación
    texto = re.sub(r"\s+", " ", texto).strip()  # Quitar espacios dobles
    return texto


df["clean_text"] = df["text"].apply(limpiar)  # Aplicar limpieza

# Paso 4️⃣: Visualizar la distribución de clases (Fake vs Real)
df["label_int"] = df["label"].map(
    {"fake": 0, "real": 1}
)  # Convertimos etiquetas a 0 y 1

# 📊 Visualización clara con Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x="label", palette="Set2")
plt.title("🧾 Distribución de clases: Fake vs Real", fontsize=14)
plt.xlabel("Etiqueta", fontsize=12)
plt.ylabel("Cantidad", fontsize=12)

# Agregamos etiquetas de conteo encima de cada barra
for p in ax.patches:
    ax.annotate(
        f"{p.get_height()}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        fontsize=12,
        color="black",
        xytext=(0, 5),
        textcoords="offset points",
    )

plt.tight_layout()
plt.show()

# Paso 5️⃣: Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label_int"],
    stratify=df["label_int"],  # Asegura que haya proporción igual de clases
    test_size=0.2,
    random_state=42,
)

# Paso 6️⃣: Vectorización de texto con tokenización subword sencilla
vectorizador = TextVectorization(
    max_tokens=1000,  # Tamaño máximo del vocabulario
    output_sequence_length=20,  # Longitud fija de cada texto tokenizado
    standardize=None,  # No volver a limpiar el texto, ya lo hicimos
    split="whitespace",  # División por espacios
)
vectorizador.adapt(X_train)  # Aprende las palabras más comunes

# Paso 7️⃣: Crear datasets optimizados para entrenamiento con tf.data
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 2

# Dataset de entrenamiento
ds_train = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

# Dataset de prueba
ds_test = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

# Paso 8️⃣: Calcular pesos de clase para corregir el desbalance (si hay más 'fake' que 'real', por ejemplo)
pesos = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(pesos))
print("Pesos de clase:", class_weight_dict)

# Paso 9️⃣: Crear el modelo neuronal ligero, ideal para CPU (no usa muchas capas ni RAM)
entrada = Input(shape=(1,), dtype=tf.string)  # Entrada de texto
x = vectorizador(entrada)  # Convertimos texto a números (tokens)
x = Embedding(input_dim=1000, output_dim=16)(x)  # Convertimos tokens a vectores densos
x = GlobalAveragePooling1D()(x)  # Promediamos los vectores de cada texto
salida = Dense(1, activation="sigmoid")(
    x
)  # Capa final con probabilidad (0=fake, 1=real)

modelo = Model(inputs=entrada, outputs=salida)
modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Paso 🔟: Callbacks que detienen el entrenamiento si ya no mejora (ahorra tiempo)
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, verbose=1),
]

# Paso 1️⃣1️⃣: Entrenamiento del modelo con pesos de clase aplicados
modelo.fit(
    ds_train,
    validation_data=ds_test,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1,
)

# Paso 1️⃣2️⃣: Guardar el modelo en formato .h5 (estándar de Keras)
modelo.save("modelo_fake_news.h5")

# Paso 1️⃣3️⃣: Convertir a formato .tflite (ligero y eficiente para CPU y móviles)
convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
modelo_tflite = convertidor.convert()

# Guardar el modelo .tflite
with open("modelo_fake_news.tflite", "wb") as f:
    f.write(modelo_tflite)

print("✅ Modelo exportado exitosamente en formatos .h5 y .tflite")
