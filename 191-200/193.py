# ⚙️ Ejercicio 193/200 — Construcción de un Pipeline seguro de inferencia para producción
# coding: utf-8
import re
import time

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, TextVectorization

# ---------- ETAPA 1: SANITIZACIÓN Y VALIDACIÓN BÁSICA ----------


def validar_texto(texto):
    texto = texto.strip().lower()

    if len(texto) < 10 or len(texto) > 300:
        return False, "Longitud inválida"

    if texto.isupper() or re.fullmatch(r"[^\w\s]+", texto):
        return False, "Texto con solo mayúsculas o símbolos"

    palabras = re.findall(r"\b\w+\b", texto)
    if len(palabras) == 0:
        return False, "Sin palabras válidas"

    for palabra in palabras:
        if re.search(r"(.)\1{3,}", palabra):
            return False, f"Repetición sospechosa: {palabra}"

    return True, "Texto válido"


# ---------- ETAPA 2: TOKENIZACIÓN SIMULADA ----------

# Tokenizador simulado con vocabulario fijo
tokenizador = TextVectorization(max_tokens=1000)
tokenizador.adapt(
    ["government fake news real cure virus pandemic election doctors truth"]
)


def procesar_texto(texto):
    secuencia = tokenizador.texts_to_sequences([texto])
    secuencia_padded = pad_sequences(
        secuencia, maxlen=20, padding="post", truncating="post"
    )
    return secuencia_padded


# ---------- ETAPA 3: MODELO SIMULADO ----------

modelo = Sequential(
    [
        Embedding(input_dim=1000, output_dim=8, input_length=20),
        GlobalAveragePooling1D(),
        Dense(1, activation="sigmoid"),
    ]
)

# ---------- ETAPA 4: PREDICCIÓN Y LOGGING ----------


def pipeline_inferencia(texto):
    valido, motivo = validar_texto(texto)
    if not valido:
        print(f"❌ Entrada rechazada: {motivo}")
        return

    entrada_procesada = procesar_texto(texto)
    pred = modelo.predict(entrada_procesada)[0][0]

    resultado = "❗FAKE" if pred >= 0.5 else "✅ REAL"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n📩 Entrada: {texto}")
    print(f"📊 Predicción: {pred:.4f} → {resultado}")
    print(f"🕒 Tiempo: {timestamp}")

    # Guardar log
    with open("inferencia_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {texto} | {pred:.4f} | {resultado}\n")


# ---------- ETAPA 5: PRUEBAS ----------

entradas = [
    "Government confirms moon landing is real",
    "FAAAAAAKE vaccine cures all!!!",
    "###@@!!!",
    "doctors discover new cure for virus",
]

for entrada in entradas:
    pipeline_inferencia(entrada)
