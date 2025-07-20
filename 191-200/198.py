# 🧠 Ejercicio 198/200 — Comparativa LSTM vs Regresión Logística con Embeddings + Reducción de Dimensión
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Input, TextVectorization

# Dataset simulado de ejemplo
textos = [
    "Government confirms the vaccine is effective",  # real
    "Scientists find new evidence of climate change",  # real
    "Click here! Shocking news about the president!",  # fake
    "You won’t believe this miracle cure for COVID!",  # fake
]
etiquetas = [1, 1, 0, 0]  # 1 = real, 0 = fake

# Limita vocabulario para no usar palabras fuera de los embeddings
vocab_size = 1000
tokenizer = TextVectorization(max_tokens=vocab_size)
tokenizer.adapt(textos)

# Convertir textos a secuencias numéricas
secuencias_padded = pad_sequences(tokenizer, padding="post")


embedding_dim = 100
ruta_glove = "glove.6B.100d.txt"  # Asegúrate de tenerlo en la carpeta

# Cargar GloVe
print("Cargando GloVe...")
embeddings_index = {}
with open(ruta_glove, encoding="utf-8") as f:
    for linea in f:
        valores = linea.split()
        palabra = valores[0]
        vectores = np.asarray(valores[1:], dtype="float32")
        embeddings_index[palabra] = vectores

# Crear matriz de embeddings
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for palabra, idx in tokenizer.word_index.items():
    if idx < vocab_size and palabra in embeddings_index:
        embedding_matrix[idx] = embeddings_index[palabra]


# Extraer embedding promedio por frase
def frase_a_embedding(frase):
    indices = tokenizer.texts_to_sequences([frase])[0]
    vectores = [embedding_matrix[idx] for idx in indices if idx < vocab_size]
    return np.mean(vectores, axis=0) if vectores else np.zeros(embedding_dim)


X = np.array([frase_a_embedding(frase) for frase in textos])

# Reducir a 20 dimensiones
pca = PCA(n_components=20)
X_reducido = pca.fit_transform(X)
y = np.array(etiquetas)

logreg = LogisticRegression()
logreg.fit(X_reducido, y)

preds_logreg = logreg.predict(X_reducido)
print("Accuracy LogReg:", accuracy_score(y, preds_logreg))

longitud_max = secuencias_padded.shape[1]

modelo_lstm = Sequential(
    [
        Input(shape=(longitud_max,)),
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=longitud_max,
            trainable=False,
        ),  # No reentrenar
        LSTM(units=16),  # Simple y ligero
        Dense(1, activation="sigmoid"),
    ]
)

modelo_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
modelo_lstm.fit(secuencias_padded, y, epochs=10, verbose=0)

_, acc_lstm = modelo_lstm.evaluate(secuencias_padded, y, verbose=0)
print("Accuracy LSTM:", acc_lstm)
