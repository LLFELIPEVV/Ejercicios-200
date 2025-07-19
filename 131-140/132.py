# ✅ Ejercicio 132/200 — Visualización profesional con matriz de confusión y curvas ROC/PR
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import (
    TextVectorization,
    Dense,
    Embedding,
    GlobalAveragePooling1D,
)

# Paso 1: Cargar datos (fake news reducido)
df_fake = pd.read_csv(r"Datasets\archive\Fake.csv")
df_true = pd.read_csv(r"Datasets\archive\True.csv")

# Agregar etiquetas
df_fake["label"] = 0
df_true["label"] = 1

# Unir y barajar
df = (
    pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
)

# Tomar una muestra pequeña para que no pese mucho
df = df.sample(n=3000, random_state=42)

# Paso 2: Preprocesamiento
texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

# Dividir en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Vectorización
vectorizer = TextVectorization(
    max_tokens=10000, output_mode="int", output_sequence_length=300
)
vectorizer.adapt(x_train)

# Crear datasets vectorizados
x_train_vect = vectorizer(np.array(x_train))
x_test_vect = vectorizer(np.array(x_test))
y_train = np.array(y_train)
y_test = np.array(y_test)

# Paso 3: Modelo simple
model = Sequential(
    [
        Embedding(input_dim=10000, output_dim=16),
        GlobalAveragePooling1D(),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(x_train_vect, y_train, epochs=5, batch_size=32, verbose=1)

# Paso 4: Evaluar modelo
y_prob = model.predict(x_test_vect).flatten()  # Probabilidades
y_pred = (y_prob >= 0.5).astype(int)  # Clases binarias

# Paso 5: Visualizaciones

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "True"],
    yticklabels=["Fake", "True"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")  # Línea base
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("pr_curve.png")
plt.close()
