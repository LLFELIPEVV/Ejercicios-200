# 🧠 Ejercicio 50/200: Clasificación binaria de texto usando GloVe + GRU (Spam vs Ham)

# 📚 Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, GRU, Dense
from keras.optimizers import Adam

# 1️⃣ Cargar y preparar los datos
df = pd.read_csv(
    "Datasets/sms+spam+collection/SMSSpamCollection", sep="\t", names=["label", "text"]
)
df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})  # Convertir etiquetas a 0 y 1

# 2️⃣ Vectorización del texto
vectorizer = TextVectorization(
    max_tokens=10000,  # Tamaño del vocabulario
    output_sequence_length=50,  # Longitud fija para todas las secuencias
    output_mode="int",  # Codificación a enteros
)
vectorizer.adapt(df["text"])  # Aprende el vocabulario del dataset

X = vectorizer(df["text"]).numpy()  # Transformar texto a secuencias numéricas
y = df["label_bin"].values  # Etiquetas binarias

# 3️⃣ Cargar embeddings preentrenados GloVe (100 dimensiones)
embedding_dim = 100
embedding_index = {}

# Cargar vectores GloVe desde archivo
with open("Gloove/glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

# 4️⃣ Crear matriz de embeddings basada en el vocabulario del vectorizador
vocab = vectorizer.get_vocabulary()
word_index = {word: idx for idx, word in enumerate(vocab)}
embedding_matrix = np.zeros((len(vocab), embedding_dim))

# Rellenar matriz con los vectores GloVe
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 5️⃣ Dividir el conjunto de datos (80% entrenamiento, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6️⃣ Definir el modelo con GRU + embeddings preentrenados
model = Sequential(
    [
        Embedding(
            input_dim=len(vocab),
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,  # No actualizamos los pesos de GloVe
        ),
        GRU(64),  # Capa GRU para procesar la secuencia
        Dense(1, activation="sigmoid"),  # Capa de salida para clasificación binaria
    ]
)

# 7️⃣ Compilar el modelo
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# 8️⃣ Entrenar el modelo
history = model.fit(
    X_train, y_train, validation_split=0.2, epochs=10, batch_size=4, verbose=1
)

# 9️⃣ Evaluar el modelo
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# 🔟 Reporte de clasificación
print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# 🔁 Matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.title("Matriz de Confusión - GloVe + GRU")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
