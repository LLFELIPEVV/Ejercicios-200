# 🧠 Ejercicio 195/200 — Comparación real: LSTM vs. modelo clásico (igualando complejidad)
# coding: utf-8
import numpy as np

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, TextVectorization

# ---------- 1. SIMULACIÓN DE DATOS ----------
# Noticias falsas vs reales (muy reducidas para prueba)
textos = [
    "The cure for cancer is hidden",  # fake
    "Aliens invaded the White House",  # fake
    "Vaccines cause autism",  # fake
    "5G spreads the virus",  # fake
    "Election was rigged by AI",  # fake
    "FDA approves new vaccine",  # real
    "Scientists develop mRNA tech",  # real
    "President addresses the nation",  # real
    "Hospitals report lower cases",  # real
    "Study confirms vaccine safety",  # real
]
etiquetas = [0] * 5 + [1] * 5  # 0 = fake, 1 = real

# División
X_train, X_test, y_train, y_test = train_test_split(
    textos, etiquetas, test_size=0.3, random_state=42
)

# ---------- 2. MODELO CLÁSICO: TF-IDF + REGRESIÓN LOGÍSTICA ----------
vectorizador = TfidfVectorizer(max_features=20)
X_train_vec = vectorizador.fit_transform(X_train)
X_test_vec = vectorizador.transform(X_test)

modelo_clasico = LogisticRegression()
modelo_clasico.fit(X_train_vec, y_train)
pred_clasico = modelo_clasico.predict(X_test_vec)

print("\n🔎 Evaluación Modelo Clásico (TF-IDF + LogisticRegression):")
print(classification_report(y_test, pred_clasico))

# ---------- 3. MODELO LSTM (≈ misma complejidad que el modelo clásico) ----------
# Tokenizamos y convertimos a secuencias
tokenizer = TextVectorization(max_tokens=50)
tokenizer.adapt(textos)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding para igualar longitud
maxlen = 5
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# LSTM pequeña, limitando los parámetros (~2000)
model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=50, output_dim=4, input_length=maxlen))  # 50x4 = 200
model_lstm.add(LSTM(units=6))  # (4+6)*6*4 = 312
model_lstm.add(Dense(1, activation="sigmoid"))  # 6+1 = 7
model_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entrenamiento
model_lstm.fit(X_train_pad, np.array(y_train), epochs=20, batch_size=1, verbose=0)

# Evaluación
pred_lstm = model_lstm.predict(X_test_pad)
pred_lstm_bin = (pred_lstm > 0.5).astype(int)

print("\n🧠 Evaluación Modelo LSTM Pequeña:")
print(classification_report(y_test, pred_lstm_bin))
