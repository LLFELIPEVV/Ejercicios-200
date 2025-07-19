# ✅ Ejercicio 134/200 — Visualización profesional: Curva ROC y Curva de Precisión-Recall en un modelo de clasificación binaria con Keras
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


# Crear un dataset sintético binario
X, y = make_classification(
    n_samples=1000, n_features=10, n_classes=2, n_informative=5, random_state=42
)

# Escalar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Construir modelo simple
model = Sequential(
    [
        Dense(16, activation="relu", input_shape=(10,)),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),  # Salida binaria con probabilidad
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

# Predecir probabilidades
y_probs = model.predict(X_test).ravel()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Curva PR
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)

# Graficar ambas
plt.figure(figsize=(12, 5))

# ROC
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC ROC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC")
plt.legend()

# PR
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f"AUC PR = {pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precisión-Recall")
plt.legend()

plt.tight_layout()
plt.show()
