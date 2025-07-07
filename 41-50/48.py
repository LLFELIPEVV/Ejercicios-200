# 🧠 Ejercicio 48/200: Clasificación de texto con GloVe + LSTM (Spam vs Ham)

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, LSTM, Dense
from keras.optimizers import Adam

# 1️⃣ Cargar dataset SMS Spam Collection
df = pd.read_csv(
    "Datasets/sms+spam+collection/SMSSpamCollection", sep="\t", names=["label", "text"]
)
df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})  # 0 = ham, 1 = spam

# 2️⃣ Vectorización del texto con una capa de Keras
vectorizer = TextVectorization(
    max_tokens=10000,  # Limita el vocabulario a 10k palabras más comunes
    output_mode="int",  # Convierte palabras a índices
    output_sequence_length=50,  # Longitud fija para todas las secuencias
)
vectorizer.adapt(df["text"])  # Aprende el vocabulario del corpus

# 3️⃣ Convertir texto a secuencias y obtener etiquetas
X = vectorizer(tf.constant(df["text"])).numpy()
y = df["label_bin"].values

# 4️⃣ Cargar vectores GloVe preentrenados (100 dimensiones)
embedding_dim = 100
embedding_index = {}  # Diccionario: palabra -> vector

with open("Gloove/glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]  # Primera palabra
        vector = np.asarray(values[1:], dtype="float32")  # Vector de 100 dimensiones
        embedding_index[word] = vector  # Agregar al diccionario

# 5️⃣ Crear matriz de embeddings alineada con el vocabulario del modelo
vocab = vectorizer.get_vocabulary()
vocab_size = len(vocab)
word_index = {word: idx for idx, word in enumerate(vocab)}  # palabra -> índice
embedding_matrix = np.zeros((vocab_size, embedding_dim))  # Inicialización

for word, idx in word_index.items():
    if idx < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector  # Insertar vector GloVe

# 6️⃣ Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7️⃣ Definir el modelo con LSTM y embeddings preentrenados
model = Sequential(
    [
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],  # Usar embeddings de GloVe
            trainable=False,  # No ajustar los vectores durante el entrenamiento
        ),
        LSTM(64),  # Capa recurrente para secuencias
        Dense(1, activation="sigmoid"),  # Salida binaria (spam vs no spam)
    ]
)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# 8️⃣ Entrenamiento del modelo
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=4)

# 9️⃣ Evaluación del modelo
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# 🔟 Matriz de Confusión
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - GloVe + LSTM")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
