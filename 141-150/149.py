# ✅ Ejercicio 149/200 — Predicción desde .json con validaciones profesionales
import os
import json
import numpy as np

from keras.models import load_model
from keras.layers import TextVectorization

# Paso 1: Validar existencia del archivo JSON
json_path = "input_news.json"
assert os.path.exists(json_path), "Archivo de entrada JSON no encontrado."

# Paso 2: Cargar el archivo JSON
try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError:
    raise ValueError("El archivo JSON tiene un formato inválido.")

# Paso 3: Validar campos esperados en el archivo
assert "title" in data and "content" in data, (
    "Faltan campos 'title' o 'content' en el JSON."
)

# Paso 4: Concatenar título + contenido como entrada de texto
text_input = data["title"] + " " + data["content"]

# Paso 5: Preprocesar texto con una capa TextVectorization entrenada
# ⚠️ Debes usar el mismo adapt() y configuración usada en el entrenamiento
# Aquí cargamos una simulación simple para mantener bajo costo

# Simulamos parámetros usados en el entrenamiento (debes mantenerlos coherentes)
max_tokens = 10000
output_sequence_length = 200

vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=output_sequence_length,
)

# Cargar vocabulario si lo tienes guardado, aquí adaptamos directamente (solo para ejemplo)
# En producción: guardar y reutilizar vocabulario con `get_vocabulary()` y `set_vocabulary()`
vectorizer.adapt([text_input])

# Vectorizamos la entrada
vectorized_input = vectorizer(np.array([text_input]))

# Paso 6: Cargar modelo previamente entrenado
model_path = "modelo_fake_news.h5"
assert os.path.exists(model_path), "El modelo .h5 no fue encontrado."

model = load_model(model_path)

# Paso 7: Realizar predicción
prediction = model.predict(vectorized_input)[0][0]

# Paso 8: Mostrar resultado con formato profesional
resultado = "NOTICIA FALSA" if prediction > 0.5 else "NOTICIA VERDADERA"
print("\n🔍 Resultado de la predicción:")
print(f"📰 ID: {data.get('id', 'sin id')}")
print(f"📖 Texto analizado: {text_input[:100]}...")
print(f"📊 Probabilidad de fake: {prediction:.2f}")
print(f"✅ Clasificación: {resultado}\n")

# Paso 9: Validación simple del sistema
assert 0.0 <= prediction <= 1.0, "Predicción fuera de rango válido [0,1]"
