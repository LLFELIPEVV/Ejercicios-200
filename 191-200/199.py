# 🧠 Ejercicio 199/200 — Detección y defensa ante ataques de texto adversarios simples
import re
import string

from keras.preprocessing.sequence import pad_sequences

textos_malos = [
    "Vaccine is ffffaaakeee!!!",  # repetición de letras + signos
    "You won’t believe thiiis...",  # elongación
    "Cure COVID $$$www.fake.com$$$",  # simbolismo + URL
    "SHOCKING NEWS!!! 100% TRUE!!!",  # mayúsculas + exageración
]


def normalize_text(texto):
    texto = texto.lower()  # todo a minúsculas
    texto = re.sub(r"http\S+", "", texto)  # eliminar URLs
    texto = re.sub(r"\d+", "", texto)  # eliminar números
    texto = re.sub(r"([" + string.punctuation + "])", r" \1 ", texto)  # separar signos
    texto = re.sub(
        r"(.)\1{2,}", r"\1", texto
    )  # eliminar letras repetidas más de 2 veces
    texto = re.sub(r"\s{2,}", " ", texto)  # espacios múltiples
    return texto.strip()


# Reusa el tokenizer, embedding_matrix y modelo del ejercicio anterior
def predecir(texto_original, model, tokenizer, longitud_max):
    # Paso 1: texto crudo
    secuencia_cruda = tokenizer.texts_to_sequences([texto_original])
    padded_cruda = pad_sequences(secuencia_cruda, maxlen=longitud_max, padding="post")

    # Paso 2: texto sanitizado
    texto_limpio = normalize_text(texto_original)
    secuencia_limpia = tokenizer.texts_to_sequences([texto_limpio])
    padded_limpia = pad_sequences(secuencia_limpia, maxlen=longitud_max, padding="post")

    # Paso 3: inferencia
    pred_cruda = model.predict(padded_cruda, verbose=0)[0][0]
    pred_limpia = model.predict(padded_limpia, verbose=0)[0][0]

    # Mostrar diferencia
    print(f"\nTexto original: {texto_original}")
    print(f"Texto limpio:   {texto_limpio}")
    print(f"Predicción cruda:  {pred_cruda:.4f}")
    print(f"Predicción limpia: {pred_limpia:.4f}")


for t in textos_malos:
    predecir(t, modelo_lstm, tokenizer, longitud_max)
