# 🧠 Ejercicio 146/200: Visualización profesional de curvas ROC y PR para modelo de detección de fake news
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from keras.models import Sequential
from keras.layers import Dense, TextVectorization

# 1. Cargar datos desde .csv
df_fake = pd.read_csv(r"Datasets\archive\Fake.csv")
df_true = pd.read_csv(r"Datasets\archive\True.csv")

# 2. Etiquetar y unir
df_fake["label"] = 0
df_true["label"] = 1
df = pd.concat([df_fake, df_true], ignore_index=True)

# 3. Validación: que existan las columnas requeridas
assert "text" in df.columns and "label" in df.columns, "Faltan columnas requeridas"

# 4. Limpieza básica: eliminar filas vacías
df = df.dropna(subset=["text", "label"])

# 5. Separar variables
X_texts = df["text"].astype(str).tolist()  # Asegurar que sean string
y_labels = df["label"].values

# 6. Tokenización simple (no entrenamos embeddings aquí)
tokenizer = TextVectorization()
tokenizer.adapt(X_texts)
X_tokens = tokenizer(tf.constant(X_texts))

# Convertir a numpy para compatibilidad con scikit-learn
X_pad = X_tokens.numpy()

# 7. División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_labels, test_size=0.2, random_state=42
)

# 8. Modelo simple
model = Sequential(
    [
        Dense(16, activation="relu", input_shape=(X_pad.shape[1],)),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

# 9. Predicción: obtener probabilidades
y_scores = model.predict(X_test).flatten()  # Probabilidades [0,1]
assert y_scores.shape[0] == y_test.shape[0], "Mismatch entre predicciones y etiquetas"

# 10. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")  # Diagonal base
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend(loc="lower right")

# 11. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.tight_layout()
plt.legend()
plt.show()
