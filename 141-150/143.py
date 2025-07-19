# 🧠 Ejercicio 143/200 – Validación del sistema: pruebas simples con assert y detección de errores
import json
import numpy as np
import tensorflow as tf

# === Paso 1: Cargar modelo entrenado (.h5) ===
try:
    model = tf.keras.models.load_model("modelo_fake_news.h5")
except OSError as e:
    print("Error al cargar el modelo:", e)
    exit()

# === Paso 2: Cargar entrada desde archivo .json ===
try:
    with open("noticia.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Archivo .json no encontrado.")
    exit()
except json.JSONDecodeError:
    print("El archivo .json no está bien formado.")
    exit()

# === Paso 3: Validar estructura del JSON ===
assert "title" in data and "text" in data, "El JSON debe tener 'title' y 'text'"

# === Paso 4: Preprocesar el texto (versión simple) ===
texto = (data["title"] + " " + data["text"]).lower().strip()

# En una implementación real, se usaría un vectorizador entrenado
# Aquí usamos una representación simple de longitud fija (placeholder)
vector_input = np.array([len(texto) % 100 / 100])  # Escala entre 0 y 1

# === Paso 5: Realizar predicción ===
try:
    resultado = model.predict(
        np.array([vector_input])
    )  # Ajustar forma según modelo real
except Exception as e:
    print("Error al predecir:", e)
    exit()

# === Paso 6: Validaciones con assert ===
assert isinstance(resultado, np.ndarray), "La predicción debe ser un ndarray"
assert resultado.shape[1] == 1, "El modelo debe producir salida binaria"
assert 0 <= resultado[0][0] <= 1, "La probabilidad debe estar entre 0 y 1"

# === Paso 7: Interpretar resultado ===
es_fake = resultado[0][0] > 0.5
print("¿Es fake news?:", "Sí" if es_fake else "No")
