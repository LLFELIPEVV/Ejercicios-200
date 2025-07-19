# 🧠 Ejercicio 142/200 — Predicción desde archivo .json + validación profesional de inputs
# predictor.py
import os
import json
import pickle
import numpy as np

from keras.models import load_model

# --- Paso 1: Verifica que los archivos existen ---
assert os.path.exists("modelo_fake_news.h5"), "❌ No se encontró el modelo .h5"
assert os.path.exists("tokenizer.pickle"), (
    "❌ No se encontró el vectorizador tokenizer.pickle"
)
assert os.path.exists("noticia.json"), "❌ No se encontró el archivo noticia.json"

# --- Paso 2: Carga el modelo entrenado ---
model = load_model("modelo_fake_news.h5")

# --- Paso 3: Carga el tokenizer ---
with open("tokenizer.pickle", "rb") as f:
    vectorizer = pickle.load(f)

# --- Paso 4: Cargar noticia desde JSON ---
with open("noticia.json", "r", encoding="utf-8") as f:
    noticia = json.load(f)

# --- Validaciones con assert ---
assert "title" in noticia, "❌ Falta campo 'title' en JSON"
assert "text" in noticia, "❌ Falta campo 'text' en JSON"
assert isinstance(noticia["text"], str), "❌ El campo 'text' debe ser string"

# Combina título + cuerpo
input_text = noticia["title"] + " " + noticia["text"]

# --- Paso 5: Preprocesar texto ---
# El vectorizador debe ser el mismo usado en el entrenamiento
vectorized_input = vectorizer(np.array([input_text]))

# --- Paso 6: Realizar la predicción ---
prediction = model.predict(vectorized_input)
label = "Fake" if prediction[0][0] >= 0.5 else "Real"

# --- Paso 7: Mostrar resultado ---
print(f"\n🔍 Resultado: {label.upper()} ({prediction[0][0]:.2f})")

# --- Paso 8: Validación profesional con assert ---
assert prediction.shape == (1, 1), "❌ La salida del modelo no es de forma (1, 1)"
assert isinstance(label, str), "❌ La predicción no generó una etiqueta válida"
