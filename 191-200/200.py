# 🧠 Ejercicio 200/200 — Pipeline de producción completo para inferencia segura de Fake News
import re
import string
import logging
import argparse
import numpy as np

from keras.models import load_model
from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences

# ========= CONFIGURACIÓN ========= #
MAX_LEN = 100  # Longitud máxima del input
MODEL_PATH = "modelo_lstm.h5"
TOKENIZER_PATH = "tokenizer_fake.npy"
EMBEDDING_PATH = "embedding_matrix.npy"
VOCAB_SIZE = 5000

# ========= LOGGER ========= #
logging.basicConfig(filename="inferencia.log", level=logging.INFO)
logger = logging.getLogger()


# ========= SANITIZACIÓN ========= #
def normalize_text(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"\d+", "", texto)
    texto = re.sub(r"([" + string.punctuation + "])", r" \1 ", texto)
    texto = re.sub(r"(.)\1{2,}", r"\1", texto)
    texto = re.sub(r"\s{2,}", " ", texto)
    return texto.strip()


# ========= CARGA TOKENIZER Y EMBEDDINGS ========= #
def cargar_tokenizer(path):
    datos = np.load(path, allow_pickle=True).item()
    tokenizer = TextVectorization(max_tokens=VOCAB_SIZE)
    tokenizer.word_index = datos
    return tokenizer


# ========= PREDICCIÓN ========= #
def predecir(texto_crudo, modelo, tokenizer):
    texto_limpio = normalize_text(texto_crudo)
    secuencia = tokenizer.texts_to_sequences([texto_limpio])
    padded = pad_sequences(secuencia, maxlen=MAX_LEN, padding="post")

    pred = modelo.predict(padded, verbose=0)[0][0]
    logger.info(f"Texto: {texto_crudo} | Limpio: {texto_limpio} | Score: {pred:.4f}")
    return pred


# ========= MAIN ========= #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detector de Fake News")
    parser.add_argument("--texto", type=str, required=True, help="Texto a analizar")
    args = parser.parse_args()

    # Cargar modelo y tokenizer
    modelo = load_model(MODEL_PATH)
    tokenizer = cargar_tokenizer(TOKENIZER_PATH)

    # Inferencia
    resultado = predecir(args.texto, modelo, tokenizer)
    print(f"\n🧠 Resultado: {resultado:.4f}")
    print(
        "💡 Interpretación: ",
        "Probable FALSA" if resultado > 0.5 else "Probable VERDADERA",
    )
