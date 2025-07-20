# 🧪 Ejercicio 187/200 — Comparación Justa: LSTM vs Modelo Clásico con Igual Complejidad
import numpy as np

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    LSTM,
    TextVectorization,
)

# 1. Datos de prueba (2 reales, 2 falsas)
corpus = [
    "Breaking news the president signs a new law",  # real
    "This just in aliens have landed in New York",  # fake
    "Government confirms economic growth of 5 percent",  # real
    "Celebrity cloned by the secret agency leaks proof",  # fake
]
etiquetas = [1, 0, 1, 0]  # 1 = real, 0 = fake

# 2. Tokenización y secuencias
tokenizer = TextVectorization(max_tokens=1000)
tokenizer.adapt(corpus)
X = pad_sequences(tokenizer, maxlen=10, padding="post")
y = np.array(etiquetas)

# 3. Modelo clásico: Embedding + Pooling + Dense
modelo_clasico = Sequential(
    [
        Embedding(input_dim=1000, output_dim=8, input_length=10),
        GlobalAveragePooling1D(),
        Dense(1, activation="sigmoid"),
    ]
)
modelo_clasico.compile(optimizer="adam", loss="binary_crossentropy")
print("\n📌 Parámetros Modelo Clásico:")
modelo_clasico.summary()

# 4. Modelo LSTM mínimo: Embedding + LSTM
modelo_lstm = Sequential(
    [
        Embedding(input_dim=1000, output_dim=8, input_length=10),
        LSTM(units=8),
        Dense(1, activation="sigmoid"),
    ]
)
modelo_lstm.compile(optimizer="adam", loss="binary_crossentropy")
print("\n📌 Parámetros Modelo LSTM:")
modelo_lstm.summary()
