# 🧠 Ejercicio 109/200 – Optimización ligera para modelos en producción: uso de .tflite + tf.data.Dataset
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import load_model
from keras.layers import TextVectorization
from sklearn.model_selection import train_test_split

model = load_model("modelo_fake_news.h5")
print("✅ Modelo cargado correctamente.")

# Convertimos a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardamos en disco
with open("modelo_fake_news.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Modelo convertido y guardado como .tflite")

# Cargar el modelo .tflite en modo inferencia
interpreter = tf.lite.Interpreter(model_path="modelo_fake_news.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Función para predecir usando el modelo .tflite
def predict_tflite(input_tensor):
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output


# Cargar datos
df = pd.concat(
    [
        pd.read_csv("Datasets/archive/Fake.csv")
        .dropna()
        .sample(500, random_state=42)
        .assign(label=0),
        pd.read_csv("Datasets/archive/True.csv")
        .dropna()
        .sample(500, random_state=42)
        .assign(label=1),
    ]
)
X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Vectorizador igual al usado en entrenamiento
vocab_size = 5000
max_len = 100
vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=max_len)
vectorizer.adapt(tf.convert_to_tensor(X_train))


# Función que transforma texto para el modelo
def vectorize(text, label):
    return vectorizer(text), label


ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
ds_test = ds_test.map(vectorize).batch(32).prefetch(tf.data.AUTOTUNE)

# Convertimos un batch a tensor numpy para predecir con .tflite
for batch_text, _ in ds_test.take(1):
    input_sample = batch_text.numpy()

# 🕐 Inferencia con modelo .tflite
start = time.time()
preds_tflite = predict_tflite(input_sample.astype(np.float32))
print("⏱️ Tiempo inferencia (.tflite):", round(time.time() - start, 4), "s")

# 🕐 Inferencia con modelo .h5
start = time.time()
preds_h5 = model.predict(input_sample)
print("⏱️ Tiempo inferencia (.h5):", round(time.time() - start, 4), "s")
