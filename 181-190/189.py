# 🧪 Ejercicio 189/200 — Sanitización de Entrada e Inferencia Segura para Modelos de Texto
import re

from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences


# Simulación de un modelo (solo para mostrar predicción segura)
def modelo_simulado(x):
    # Valor ficticio entre 0 y 1
    return 0.42


# Tokenizador simulado entrenado sobre corpus previo
tokenizer = TextVectorization(max_tokens=1000)
corpus = [
    "government confirms vaccine news",
    "fake news alert",
    "breaking hoax",
    "economy grows fast",
]
tokenizer.adapt(corpus)

# ----------------------
# Función de sanitización
# ----------------------


def sanitizar_y_predecir(texto_usuario):
    print("📥 Entrada original:", repr(texto_usuario))

    # 1. Limpiar espacios y bajar a minúsculas
    texto = texto_usuario.strip().lower()

    # 2. Detectar si está vacío
    if len(texto) == 0:
        return "⚠️ Texto vacío. Intenta escribir una noticia real."

    # 3. Eliminar símbolos raros repetidos
    texto_limpio = re.sub(r"[^a-zA-Z0-9\s]", "", texto)  # quita emojis, signos
    palabras = texto_limpio.split()

    # 4. Validación: al menos 2 palabras
    if len(palabras) < 2:
        return "⚠️ Texto muy corto o sin palabras válidas."

    # 5. Detección de spam (palabras repetidas >30%)
    repeticiones = sum(palabras.count(p) > 1 for p in set(palabras))
    if repeticiones / len(palabras) > 0.3:
        return "⚠️ Texto sospechoso: demasiadas repeticiones."

    # 6. Detección de caracteres especiales (>70%)
    if (
        len(re.findall(r"[^a-zA-Z0-9\s]", texto_usuario)) / max(1, len(texto_usuario))
        > 0.7
    ):
        return "⚠️ Entrada inválida: contiene demasiados símbolos raros."

    # 7. Todo bien → convertir a secuencia y predecir
    secuencia = tokenizer.texts_to_sequences([texto])
    secuencia_padded = pad_sequences(secuencia, maxlen=10)

    resultado = modelo_simulado(secuencia_padded)
    return f"✅ Predicción simulada: probabilidad de noticia real = {resultado:.2f}"


# ----------------------
# Pruebas reales
# ----------------------

entradas = [
    "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥",
    "real news government update",
    "!!! !!! !!!",
    "this this this this",
    "   ",
    "covid confirmed by WHO",
    "<script>alert(1)</script>",
]

for entrada in entradas:
    print(sanitizar_y_predecir(entrada))
    print("-" * 60)
