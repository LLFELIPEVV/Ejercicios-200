# 🧪 Ejercicio 114/200 — Explicabilidad con LIME en un modelo liviano de texto
# 🔧 Importamos librerías necesarias
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from lime.lime_text import LimeTextExplainer

# 1️⃣ Dataset simulado: frases y etiquetas (1=fake, 0=real)
data = {
    "text": [
        "Breaking news: President caught lying!",
        "NASA discovers water on Mars!",
        "You won't believe what this dog did...",
        "Click here to win $1000!",
        "Scientists prove Earth is round.",
        "Miracle cure for baldness found!",
        "Government hides the truth again!",
        "Facebook down for 3 hours.",
    ],
    "label": [1, 0, 1, 1, 0, 1, 1, 0],
}
df = pd.DataFrame(data)


# 2️⃣ Limpiamos el texto (minúsculas, sin signos, sin URLs)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


df["text"] = df["text"].apply(clean_text)

# 3️⃣ Separación en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42
)

# 4️⃣ Vectorización del texto a secuencias numéricas
vectorizer = TextVectorization(
    max_tokens=1000, output_mode="int", output_sequence_length=20
)
vectorizer.adapt(X_train)

# 5️⃣ Preparación eficiente con tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))


def vectorize(text, label):
    return vectorizer(text), label


train_ds = train_ds.map(vectorize).batch(2).cache().prefetch(1)
test_ds = test_ds.map(vectorize).batch(2).cache().prefetch(1)

# 6️⃣ Modelo sencillo y ligero
model = Sequential(
    [
        Embedding(input_dim=1000, output_dim=16),
        GlobalAveragePooling1D(),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 7️⃣ Entrenamiento rápido
model.fit(train_ds, validation_data=test_ds, epochs=10, verbose=0)

# 8️⃣ Creamos función que LIME necesita (recibe texto y devuelve predicción)
class_names = ["real", "fake"]


def predict_fn(texts):
    # Convierte texto sin procesar en secuencias y predice con el modelo
    sequences = vectorizer(tf.constant(texts))
    preds = model.predict(sequences)
    return np.hstack((1 - preds, preds))  # Probabilidades clase 0 y 1


# 9️⃣ LIME: explicamos una noticia del conjunto de prueba
explainer = LimeTextExplainer(class_names=class_names)

i = 0  # índice del texto a explicar
text_to_explain = X_test.iloc[i]
print(f"\n📝 Texto a explicar:\n{text_to_explain}")

exp = explainer.explain_instance(text_to_explain, predict_fn, num_features=6)

# 🔟 Mostramos explicación como gráfico
weights = dict(exp.as_list())
words = list(weights.keys())
scores = list(weights.values())

# Gráfico con Seaborn
sns.set_style("whitegrid")
plt.figure(figsize=(8, 4))
sns.barplot(x=scores, y=words, palette="viridis")
plt.title("Palabras que influyeron en la predicción (LIME)")
plt.xlabel("Contribución al modelo")
plt.ylabel("Palabra")
plt.tight_layout()
plt.show()
