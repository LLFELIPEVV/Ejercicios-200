# 🧠 Ejercicio 47/200: Clasificación binaria de texto usando GloVe (Spam vs Ham)
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, Flatten, Dense
from keras.optimizers import Adam

# 1️⃣ Cargar dataset SMS Spam Collection
df = pd.read_csv(
    "Datasets/sms+spam+collection/SMSSpamCollection", sep="\t", names=["label", "text"]
)
df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})

# 2️⃣ Vectorización del texto
vectorizer = TextVectorization(
    max_tokens=10000,  # Tamaño máximo del vocabulario
    output_mode="int",  # Convierte cada palabra a un entero
    output_sequence_length=50,  # Longitud fija de cada texto
)
vectorizer.adapt(df["text"])  # Aprende el vocabulario

X = vectorizer(tf.constant(df["text"])).numpy()
y = df["label_bin"].values

# 3️⃣ Cargar vectores preentrenados GloVe (100 dimensiones)
embedding_dim = 100
embedding_index = {}

with open("Gloove/glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

# 4️⃣ Crear matriz de embeddings (alineada al vocabulario usado)
vocab = vectorizer.get_vocabulary()
word_index = {word: idx for idx, word in enumerate(vocab)}
embedding_matrix = np.zeros((10000, embedding_dim))

for word, i in word_index.items():
    if i < 10000:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # Solo si existe en GloVe

# 5️⃣ División en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6️⃣ Definir modelo con Embedding preentrenado + capas densas
model = Sequential(
    [
        Embedding(
            input_dim=10000,
            output_dim=embedding_dim,
            weights=[embedding_matrix],  # Importante: se pasan como lista
            input_length=50,
            trainable=False,  # No se actualizan durante el entrenamiento
        ),
        Flatten(),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),  # Para clasificación binaria
    ]
)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# 7️⃣ Entrenamiento del modelo
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=4)

# 8️⃣ Evaluación
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# 9️⃣ Matriz de Confusión
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - GloVe")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()
