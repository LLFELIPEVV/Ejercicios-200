# 🧠 Ejercicio 110/200 — Visualización de inferencias con LIME + Preparación óptima de datos con tf.data.Dataset
# Paso 1: Cargar librerías necesarias
import time
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import (
    TextVectorization,
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
)
from lime.lime_text import LimeTextExplainer

# Paso 2: Creamos un conjunto de datos de ejemplo muy pequeño
texts = [
    "El gobierno anunció nuevas medidas económicas.",
    "Una invasión extraterrestre está en camino.",
    "Expertos aprueban la vacuna contra el virus.",
    "El mundo será destruido mañana, dicen científicos falsos.",
    "Nuevas evidencias confirman el cambio climático.",
    "Celebridad dice haber viajado al futuro.",
]
labels = [0, 1, 0, 1, 0, 1]  # 0: noticia real, 1: fake news

# Paso 3: Convertimos a Dataset eficiente
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 2

# Creamos Dataset de entrenamiento
ds = tf.data.Dataset.from_tensor_slices((texts, labels))
ds = ds.shuffle(buffer_size=6).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Paso 4: Capa de vectorización del texto
vectorizer = TextVectorization(
    max_tokens=1000, output_mode="int", output_sequence_length=20
)
vectorizer.adapt(ds.map(lambda x, y: x))  # Solo extrae los textos

# Paso 5: Crear modelo secuencial sencillo
model = Sequential(
    [
        vectorizer,  # convierte texto a secuencia de enteros
        Embedding(input_dim=1000, output_dim=16),
        GlobalAveragePooling1D(),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),  # salida binaria
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(ds, epochs=10, verbose=0)

# Guardamos como .h5
model.save("modelo_noticias.h5")
print("✅ Modelo guardado como .h5")

# Paso 6: Convertimos a .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("modelo_noticias.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Modelo convertido y guardado como .tflite")


# Paso 7: Inferencia (comparación de tiempos)
def predict_text_tflite(text):
    # Vectorizamos manualmente
    input_tensor = vectorizer(tf.constant([text]))  # (1, seq_len)
    input_tensor = tf.cast(input_tensor, tf.float32)

    interpreter = tf.lite.Interpreter(model_path="modelo_noticias.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_tensor.numpy())
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    return prediction[0][0]


# Medimos tiempo
start = time.time()
pred_tflite = predict_text_tflite("Los aliens llegaron ayer a Colombia.")
end = time.time()
print(f"⏱️ Tiempo inferencia (.tflite): {end - start:.4f} s")

# Inferencia con modelo original
start = time.time()
pred_h5 = model.predict(["Los aliens llegaron ayer a Colombia."], verbose=0)
end = time.time()
print(f"⏱️ Tiempo inferencia (.h5): {end - start:.4f} s")


# Paso 8: LIME para explicabilidad
class FakeNewsClassifier:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def predict_proba(self, texts):
        # Devuelve probabilidad de clase 1 (fake news)
        predictions = self.model.predict(texts, verbose=0)
        return np.hstack([1 - predictions, predictions])


explainer = LimeTextExplainer(class_names=["Real", "Fake"])
wrapped_model = FakeNewsClassifier(model, vectorizer)

exp = explainer.explain_instance(
    "Los aliens llegaron ayer a Colombia.", wrapped_model.predict_proba, num_features=5
)
