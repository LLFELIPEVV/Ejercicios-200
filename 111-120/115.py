# 🧪 Ejercicio 115/200 — Explicabilidad ligera con LIME en modelos de texto optimizados
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense

# Importar LIME (puede necesitar instalación: pip install lime)
from lime.lime_text import LimeTextExplainer

# 1. Datos simulados de noticias
data = {
    "text": [
        "NASA confirms alien life on Mars!",
        "Local bakery wins international award",
        "Cure for cancer discovered!",
        "City council increases budget for education",
    ],
    "label": [1, 0, 1, 0],  # 1: Fake, 0: Real
}
df = pd.DataFrame(data)


# 2. Limpieza básica del texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


df["text"] = df["text"].apply(clean_text)

# 3. División de datos
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.5, random_state=42
)

# 4. TextVectorization
vectorizer = TextVectorization(
    max_tokens=1000, output_mode="int", output_sequence_length=20
)
vectorizer.adapt(X_train)


# 5. Dataset optimizado para CPU
def vectorize_text(text, label):
    return vectorizer(text), label


train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(vectorize_text)
    .batch(2)
    .cache()
    .prefetch(1)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .map(vectorize_text)
    .batch(2)
    .cache()
    .prefetch(1)
)

# 6. Modelo liviano
model = Sequential(
    [Embedding(1000, 16), GlobalAveragePooling1D(), Dense(1, activation="sigmoid")]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 7. Entrenamiento con EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
model.fit(
    train_ds, validation_data=test_ds, epochs=20, callbacks=[early_stop], verbose=1
)


# 8. Clase wrapper para LIME (porque necesita texto original como entrada)
class KerasFakeNewsWrapper:
    def __init__(self, keras_model, vectorizer):
        self.model = keras_model
        self.vectorizer = vectorizer

    def predict(self, texts):
        vect_texts = tf.constant(texts)
        vect = self.vectorizer(vect_texts)
        preds = self.model(vect).numpy().flatten()
        return np.stack([1 - preds, preds], axis=1)  # Probabilidades para [Real, Fake]


# 9. Instanciar LIME
explainer = LimeTextExplainer(class_names=["Real", "Fake"])
wrapped_model = KerasFakeNewsWrapper(model, vectorizer)

# 10. Elegimos un texto del test para explicar
sample_text = X_test.iloc[0]
print("Texto a analizar:", sample_text)

# 11. Generamos explicación con LIME
exp = explainer.explain_instance(sample_text, wrapped_model.predict, num_features=6)
weights = dict(exp.as_list())

# 12. Visualizamos con Seaborn
plt.figure(figsize=(8, 4))
sns.barplot(x=list(weights.values()), y=list(weights.keys()), palette="viridis")
plt.title("Importancia de palabras según LIME")
plt.xlabel("Peso en la predicción (positiva = Fake)")
plt.ylabel("Palabra")
plt.tight_layout()
plt.show()
