# 🧠 Ejercicio 152/200 — Organización profesional de scripts .py para clasificación de texto
# 🧱 Estructura del proyecto sugerida:
# fake_news_project/
# │
# ├── main.py
# ├── clean_text.py
# ├── load_data.py
# ├── train_model.py
# ├── predict.py
# ├── utils.py
# ├── data/
# │   ├── news.csv
# │   └── input_sample.txt
# └── models/
#    └── model.h5

# 1. clean_text.py – Funciones para limpiar el texto
import re
import string


def clean_text(text):
    """
    Realiza limpieza básica del texto:
    elimina signos de puntuación, convierte a minúsculas y remueve espacios extra.
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2. load_data.py – Función para cargar y limpiar el dataset
import pandas as pd
from clean_text import clean_text


def load_and_prepare_data(csv_path):
    """
    Carga un archivo CSV con columnas 'text' y 'label'.
    Limpia duplicados e inconsistencias de texto.
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="text")
    df["text"] = df["text"].fillna("").apply(clean_text)
    df["label"] = df["label"].astype(int)
    return df["text"].tolist(), df["label"].tolist()


# 3. train_model.py – Entrenamiento de un modelo simple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os


def train_model(texts, labels, save_path):
    """
    Entrena un modelo simple y lo guarda en la carpeta models/
    """
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, padding="post", maxlen=100)

    model = Sequential(
        [
            Embedding(1000, 16, input_length=100),
            GlobalAveragePooling1D(),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(
        padded,
        np.array(labels),
        epochs=10,
        batch_size=32,
        callbacks=[EarlyStopping(monitor="loss", patience=2)],
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save(os.path.join(save_path, "model.h5"))

    return tokenizer


# 4. predict.py – Predicción desde archivo .txt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from clean_text import clean_text


def predict_from_txt(txt_path, tokenizer, model_path):
    """
    Predice si un texto en un archivo `.txt` es real o falso.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text = clean_text(raw_text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding="post")

    model = load_model(model_path)
    prob = model.predict(padded)[0][0]
    return "Real" if prob > 0.5 else "Fake"


# 5. main.py – Script principal para entrenamiento y predicción
from load_data import load_and_prepare_data
from train_model import train_model
from predict import predict_from_txt

if __name__ == "__main__":
    # Entrenamiento
    texts, labels = load_and_prepare_data("data/news.csv")
    tokenizer = train_model(texts, labels, save_path="models")

    # Predicción
    prediction = predict_from_txt("data/input_sample.txt", tokenizer, "models/model.h5")
    print(f"La noticia es probablemente: {prediction}")
