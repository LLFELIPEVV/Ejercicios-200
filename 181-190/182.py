# 🎯 Ejercicio 182/200 — Sanitización de entrada e inferencia segura en modelos de texto
import re
import html
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


# Paso 1: Simulamos un modelo entrenado (dummy)
# En producción esto sería reemplazado por un modelo real
def modelo_dummy(vector):
    # Regla arbitraria: si hay muchas palabras raras, clasificar como "fake"
    rareza = np.sum(vector.toarray() < 0.01)
    return "Fake" if rareza > 5 else "Real"


# Paso 2: Sanitización robusta del texto
def sanitizar(texto):
    # Convertimos HTML en texto legible
    texto = html.unescape(texto)

    # Eliminamos etiquetas HTML comunes
    texto = re.sub(r"<[^>]+>", "", texto)

    # Eliminamos caracteres que no sean letras o espacios
    texto = re.sub(r"[^a-zA-Z\s]", " ", texto)

    # Reemplazamos múltiples espacios por uno solo
    texto = re.sub(r"\s+", " ", texto).strip()

    # Minúsculas
    texto = texto.lower()

    return texto


# Paso 3: Detección básica de entrada maliciosa
def es_maliciosa(texto):
    # Heurísticas simples
    if len(texto) < 10:
        return True
    if texto.count("http") > 2:
        return True
    if any(
        palabra in texto for palabra in ["<script", "drop table", "alert(", "onclick"]
    ):
        return True
    if len(set(texto.split())) < 3:
        return True  # muchas repeticiones
    return False


# Paso 4: Pipeline de inferencia
def inferencia_segura(texto, vectorizador):
    if es_maliciosa(texto):
        return "⚠️ Entrada rechazada por sospecha de contenido malicioso."

    limpio = sanitizar(texto)
    vector = vectorizador.transform([limpio])
    prediccion = modelo_dummy(vector)
    return f"🧠 Predicción: {prediccion}"


# Paso 5: Vectorizador preajustado (como si ya tuviera un vocabulario)
corpus_de_ejemplo = [
    "the government confirms the report is accurate and based on facts",
    "click here to win a free iphone now",
    "vaccines are a hoax and the truth is hidden",
]
vectorizador = TfidfVectorizer(max_features=100)
vectorizador.fit([sanitizar(texto) for texto in corpus_de_ejemplo])

# Paso 6: Entrada por consola
if __name__ == "__main__":
    print("=== Sistema de Inferencia Segura para Fake News ===\n")
    entrada = input("Introduce una noticia: ")
    resultado = inferencia_segura(entrada, vectorizador)
    print(resultado)
