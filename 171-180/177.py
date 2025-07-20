# 🧠 Ejercicio 177/200 — Detección y sanitización de entradas maliciosas simples en modelos de texto
import string
import numpy as np

from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences


# 👉 Simulamos un modelo ya entrenado
# ⚠️ En un entorno real usarías load_model("ruta.h5")
class ModeloSimulado:
    def predict(self, x):
        return np.array([[0.8]]) if "president" in x else np.array([[0.2]])


modelo = ModeloSimulado()

# 👉 Tokenizador simulado
tokenizer = TextVectorization(max_tokens=1000)
tokenizer.adapt(
    [
        "The president visited the disaster area and promised aid",
        "Aliens have landed and taken over the White House",
    ]
)


# Paso 1: Sanitizar texto
def sanitize_input(texto):
    # Eliminar puntuación
    texto = texto.translate(str.maketrans("", "", string.punctuation))

    # Convertir a minúsculas
    texto = texto.lower()

    # Eliminar palabras que contengan menos de 2 letras o no sean alfabéticas
    palabras = texto.split()
    palabras = [p for p in palabras if len(p) > 1 and p.isalpha()]

    # Eliminar repeticiones consecutivas simples (ej: "fake fake fake")
    texto_sin_repeticiones = []
    anterior = ""
    for palabra in palabras:
        if palabra != anterior:
            texto_sin_repeticiones.append(palabra)
            anterior = palabra

    return " ".join(texto_sin_repeticiones)


# Paso 2: Pipeline completo de inferencia segura
def predict_safe(texto_entrada):
    texto_limpio = sanitize_input(texto_entrada)
    secuencia = tokenizer.texts_to_sequences([texto_limpio])
    secuencia_pad = pad_sequences(secuencia, maxlen=10)
    return modelo.predict(secuencia_pad)[0][0]


# 📌 Ejemplos de prueba
entradas = [
    "FREEDOM freedom freedom freedom 4g bleach kill vaccine",
    "ThegoverNNNmentisFAKE!!!",
    "Aliens have landed and taken over the White House",
]

for i, texto in enumerate(entradas):
    resultado = predict_safe(texto)
    print(f"Entrada {i + 1}: {texto}")
    print(f"→ Sanitizada: {sanitize_input(texto)}")
    print(f"→ Predicción segura: {resultado:.3f}\n")
