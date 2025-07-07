# 🧠 Ejercicio 49/200: Clasificación binaria con embeddings GloVe y Bidirectional LSTM
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense
from keras.optimizers import Adam

# 1️⃣ Cargar y preparar el dataset
df = pd.read_csv(
    "Datasets/sms+spam+collection/SMSSpamCollection", sep="\t", names=["label", "text"]
)
df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})  # Mapear etiquetas a 0 y 1

# 2️⃣ Vectorización del texto: convierte el texto en secuencias de enteros
vectorizer = TextVectorization(
    max_tokens=10000, output_sequence_length=50, output_mode="int"
)
vectorizer.adapt(df["text"])  # Aprende el vocabulario a partir del dataset

X = vectorizer(tf.constant(df["text"])).numpy()
y = df["label_bin"].values

# 3️⃣ Cargar los vectores preentrenados de GloVe (100 dimensiones)
embedding_dim = 100
embedding_index = {}

with open("Gloove/glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

# 4️⃣ Crear la matriz de embeddings alineada con el vocabulario usado por TextVectorization
vocab = vectorizer.get_vocabulary()
word_index = {word: idx for idx, word in enumerate(vocab)}

embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 5️⃣ Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6️⃣ Definir el modelo con capa Embedding no entrenable + Bidirectional LSTM
model = Sequential(
    [
        Embedding(
            input_dim=len(vocab),
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,  # No entrenar los embeddings preentrenados
        ),
        Bidirectional(LSTM(64)),
        Dense(1, activation="sigmoid"),  # Salida binaria: 0 o 1
    ]
)

# 7️⃣ Compilar el modelo
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# 8️⃣ Entrenar el modelo
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=4)

# 9️⃣ Evaluar el modelo
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# 🔟 Reporte de clasificación
print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# 1️⃣1️⃣ Matriz de Confusión
plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.title("📊 Matriz de Confusión - GloVe + Bidirectional LSTM")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
