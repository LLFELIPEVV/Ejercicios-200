# 🧠 Ejercicio 138/200 — Distilación de Modelo Ligero para Fake News
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from keras.models import Sequential
from keras.layers import (
    Dense,
    TextVectorization,
    Embedding,
    GlobalAveragePooling1D,
    Dropout,
)

# Paso 1: Cargar datos
df_fake = pd.read_csv(r"Datasets\archive\Fake.csv")
df_true = pd.read_csv(r"Datasets\archive\True.csv")

# Etiquetar
df_fake["label"] = 0
df_true["label"] = 1

# Unir y mezclar
df = (
    pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
)


# Paso 2: Limpieza básica con regex
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)  # eliminar urls
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)  # quitar números y puntuación
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


df["text"] = df["text"].astype(str).apply(limpiar_texto)

# Paso 3: División del dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Paso 4: TextVectorization
vectorizer = TextVectorization(output_sequence_length=100, max_tokens=10000)
vectorizer.adapt(X_train)

X_train_vec = vectorizer(X_train)
X_test_vec = vectorizer(X_test)

# Paso 5: Modelo Maestro (más pesado)
maestro = Sequential(
    [
        Embedding(10000, 16),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)
maestro.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

maestro.fit(X_train_vec, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Paso 6: Obtener soft labels
soft_labels = maestro.predict(X_train_vec)

# Paso 7: Modelo Estudiante (más ligero)
estudiante = Sequential(
    [
        Embedding(10000, 16),
        GlobalAveragePooling1D(),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)
estudiante.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entrenamos el estudiante con las soft labels del maestro
estudiante.fit(X_train_vec, soft_labels, epochs=5, batch_size=32)

# Paso 8: Evaluar y comparar
y_pred_est = estudiante.predict(X_test_vec).flatten()
y_pred_maestro = maestro.predict(X_test_vec).flatten()

# ROC AUC
auc_est = roc_auc_score(y_test, y_pred_est)
auc_maestro = roc_auc_score(y_test, y_pred_maestro)
print(f"AUC Maestro: {auc_maestro:.4f} | AUC Estudiante: {auc_est:.4f}")

# Matriz de Confusión (con umbral 0.5)
cm = confusion_matrix(y_test, y_pred_est > 0.5)
ConfusionMatrixDisplay(cm).plot()
plt.title("Matriz de Confusión - Estudiante")
plt.show()

# Curva ROC
fpr1, tpr1, _ = roc_curve(y_test, y_pred_maestro)
fpr2, tpr2, _ = roc_curve(y_test, y_pred_est)

plt.plot(fpr1, tpr1, label="Maestro")
plt.plot(fpr2, tpr2, label="Estudiante")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC")
plt.legend()
plt.grid()
plt.show()
