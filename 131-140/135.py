# 🧠 Ejercicio 135/200 — Visualización profesional con curva ROC y matriz de confusión
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

# 1. Generar datos binarios simulados (simulan noticias falsas o verdaderas)
X, y = make_classification(
    n_samples=1000, n_features=10, weights=[0.6, 0.4], random_state=42
)

# 2. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Modelo pequeño y eficiente
model = Sequential(
    [
        Dense(16, activation="relu", input_shape=(10,)),
        Dense(1, activation="sigmoid"),  # Salida binaria
    ]
)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# 4. Obtener predicciones y probabilidades
y_pred_prob = model.predict(X_test).ravel()  # Probabilidades reales
y_pred_class = (y_pred_prob > 0.5).astype(int)  # Etiquetas binarias

# 5. Matriz de confusión
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"],
)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()

# 6. Curva ROC y AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
