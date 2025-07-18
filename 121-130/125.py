# ✅ Ejercicio 125/200 — Limpieza avanzada + Tokenización personalizada con TextVectorization + Visualización de clases
# ✅ Paso 1: Importar librerías necesarias
# Estas son herramientas que permiten procesar texto, visualizar datos y crear modelos de Machine Learning.
import re  # para limpiar texto con expresiones regulares
import numpy as np  # para trabajar con arreglos numéricos
import pandas as pd  # para trabajar con datos en formato de tabla
import seaborn as sns  # para gráficos bonitos y profesionales
import matplotlib.pyplot as plt  # para mostrar gráficos
import tensorflow as tf  # para crear y entrenar redes neuronales

from sklearn.model_selection import train_test_split  # para dividir los datos
from sklearn.utils.class_weight import (
    compute_class_weight,
)  # para manejar clases desbalanceadas

# Desde Keras (parte de TensorFlow) importamos funciones necesarias para el modelo
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    Dense,
    GlobalAveragePooling1D,
    TextVectorization,
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ✅ Paso 2: Creamos un pequeño conjunto de datos simulados con ejemplos "fake" y "real"
data = {
    "text": [
        "Breaking news: real thing!",
        "Clickbait! fake click here!",
        "Confirmed by experts",
        "Aliens??!! FAKE!",
        "Tea helps health",
        "NASA hides truth",
        "Confirmed theory",
        "HOAX again",
        "President speaks",
        "Banana cure fake",
        "WHO confirms",
        "Lies and fakes",
        "Stable economy",
        "Aliens again!",
        "Crops benefit from rain",
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
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
    ],
}
df = pd.DataFrame(data)  # Convertimos los datos en una tabla


# ✅ Paso 3: Función para limpiar el texto con expresiones regulares
# Esta función elimina todo lo que puede confundir al modelo: mayúsculas, URLs, números y signos.
def limpiar_texto(texto):
    texto = texto.lower()  # convierte todo a minúsculas
    texto = re.sub(r"http\S+", "", texto)  # elimina URLs
    texto = re.sub(r"\d+", "", texto)  # elimina números
    texto = re.sub(r"[^a-z\s]", "", texto)  # elimina signos y puntuación
    texto = re.sub(r"\s+", " ", texto).strip()  # elimina espacios duplicados
    return texto


df["clean_text"] = df["text"].apply(limpiar_texto)  # Aplicamos la limpieza a cada fila

# ✅ Paso 4: Visualizamos la cantidad de ejemplos de cada clase (fake vs real)
df["label_int"] = df["label"].map({"fake": 0, "real": 1})  # Convertimos a números

# Seaborn hace un gráfico de barras para ver si hay desbalance en las clases
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x="label", palette="pastel", edgecolor="black")
plt.title("Distribución de clases: Fake vs Real", fontsize=14)
plt.xlabel("Clase", fontsize=12)
plt.ylabel("Número de ejemplos", fontsize=12)
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + 0.3, p.get_height() + 0.2))
plt.tight_layout()
plt.show()

# ✅ Paso 5: Dividimos el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],  # textos limpios
    df["label_int"],  # etiquetas numéricas
    test_size=0.2,  # usamos 20% para pruebas
    stratify=df["label_int"],  # mantiene el balance en ambas divisiones
    random_state=42,
)

# ✅ Paso 6: Preparamos la capa de vectorización personalizada
vectorizador = TextVectorization(
    max_tokens=1000,  # máximo de palabras únicas que se guardarán
    output_sequence_length=20,  # largo uniforme de cada texto
    standardize=None,  # ya hicimos la limpieza, así que no es necesaria aquí
)
vectorizador.adapt(X_train)  # El vectorizador aprende del texto

# ✅ Paso 7: Convertimos los datos a formato eficiente tf.data.Dataset
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 4  # número de textos procesados al mismo tiempo

ds_train = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)
ds_test = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

# ✅ Paso 8: Calculamos pesos para equilibrar las clases (porque hay más "real" que "fake")
pesos = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(pesos))
print("Pesos de clase calculados:", class_weight_dict)

# ✅ Paso 9: Construimos un modelo compacto ideal para CPU
entrada = Input(shape=(1,), dtype=tf.string)  # entrada: texto sin procesar
x = vectorizador(entrada)  # aplica tokenización
x = Embedding(input_dim=1000, output_dim=16)(x)  # convierte palabras a vectores
x = GlobalAveragePooling1D()(x)  # hace un promedio para reducir dimensiones
salida = Dense(1, activation="sigmoid")(
    x
)  # salida entre 0 y 1 (probabilidad de fake/real)

modelo = Model(entrada, salida)  # creamos el modelo final
modelo.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)  # configuramos el entrenamiento

# ✅ Paso 🔟: Callbacks para entrenamiento inteligente
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

# ✅ Paso 1️⃣1️⃣: Entrenamos el modelo
modelo.fit(
    ds_train,
    validation_data=ds_test,
    epochs=20,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict,  # usamos los pesos de clase calculados
    verbose=1,
)
