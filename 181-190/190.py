# 🧪 Ejercicio 190/200 — Comparación controlada: Naive Bayes vs. LSTM con embeddings simples de Keras
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, TextVectorization

# 1. Dataset simulado (4 fake, 4 reales)
textos = [
    "Breaking: vaccine causes autism",  # Fake
    "Click here to win a free iPhone",  # Fake
    "Aliens built the pyramids",  # Fake
    "COVID was created in a lab",  # Fake
    "WHO approves new vaccine",  # Real
    "Economy grows faster this quarter",  # Real
    "Elections were held peacefully",  # Real
    "NASA confirms water on Mars",  # Real
]
etiquetas = [0, 0, 0, 0, 1, 1, 1, 1]  # 0 = Fake, 1 = Real

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    textos, etiquetas, test_size=0.25, random_state=42
)

# ----------------------------
# 2. MODELO CLÁSICO: Naive Bayes
# ----------------------------

# Usamos conteo de palabras (bolsa de palabras)
vectorizador = CountVectorizer()
X_train_vec = vectorizador.fit_transform(X_train)
X_test_vec = vectorizador.transform(X_test)

modelo_nb = MultinomialNB()
modelo_nb.fit(X_train_vec, y_train)

pred_nb = modelo_nb.predict(X_test_vec)
acc_nb = accuracy_score(y_test, pred_nb)
print(f"🔍 Naive Bayes accuracy: {acc_nb:.2f}")

# ----------------------------
# 3. MODELO LSTM SIMPLE
# ----------------------------

# Tokenizar texto para embedding
tokenizer = TextVectorization(max_tokens=1000)
tokenizer.adapt(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding a longitud fija (para LSTM)
X_train_pad = pad_sequences(X_train_seq, maxlen=10)
X_test_pad = pad_sequences(X_test_seq, maxlen=10)

# Arquitectura mínima: Embedding + LSTM + Dense
modelo_lstm = Sequential(
    [
        Embedding(input_dim=1000, output_dim=8, input_length=10),
        LSTM(8),  # Número mínimo de neuronas
        Dense(1, activation="sigmoid"),
    ]
)

modelo_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
modelo_lstm.fit(X_train_pad, np.array(y_train), epochs=10, verbose=0)

_, acc_lstm = modelo_lstm.evaluate(X_test_pad, np.array(y_test), verbose=0)
print(f"🔍 LSTM accuracy: {acc_lstm:.2f}")
