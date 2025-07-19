# 🧠 Ejercicio 139/200 – Limpieza semi-automatizada y visualización avanzada del rendimiento
# -------------------------------
# Paso 1: Lectura y Limpieza
# -------------------------------
import re
import string
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from keras.models import Sequential
from keras.layers import TextVectorization
from keras.layers import Embedding, GlobalAveragePooling1D, Dense


# Cargar CSV
df_fake = pd.read_csv(r"Datasets\archive\Fake.csv")
df_true = pd.read_csv(r"Datasets\archive\True.csv")

# Etiquetar
df_fake["label"] = 0
df_true["label"] = 1

# Unir y mezclar
df = (
    pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
)

# Verificar estado de los datos
print("Nulos por columna:\n", df.isna().sum())

# Limpieza básica semi-automatizada
df = df.dropna().drop_duplicates()

# Unir título y texto
df["content"] = df["title"] + " " + df["text"]


# Función para limpieza textual
def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(f"[{re.escape(string.punctuation)}]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


df["content"] = df["content"].apply(limpiar)

# -------------------------------
# Paso 2: Preparación con Keras
# -------------------------------
X = df["content"].values
y = df["label"].values  # 0 = real, 1 = fake

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=100)
vectorizer.adapt(X_train)

# Convertir a tensores
X_train_vec = vectorizer(X_train)
X_test_vec = vectorizer(X_test)

# -------------------------------
# Paso 3: Modelo base simple
# -------------------------------
modelo = Sequential(
    [
        Embedding(input_dim=10000, output_dim=16),
        GlobalAveragePooling1D(),
        Dense(1, activation="sigmoid"),
    ]
)

modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
modelo.fit(X_train_vec, y_train, epochs=5, validation_data=(X_test_vec, y_test))

# -------------------------------
# Paso 4: Visualización avanzada
# -------------------------------
# Predicciones para evaluación
y_pred_prob = modelo.predict(X_test_vec).ravel()
y_pred_bin = (y_pred_prob >= 0.5).astype(int)

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_bin)
ConfusionMatrixDisplay(cm).plot()
plt.title("Matriz de Confusión")
plt.show()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.title("Curva ROC")
plt.xlabel("Falsos Positivos")
plt.ylabel("Verdaderos Positivos")
plt.grid(True)
plt.show()

# Curva Precision-Recall
prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
plt.plot(rec, prec)
plt.title("Curva Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()
