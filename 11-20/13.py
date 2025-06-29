# Ejercicio 13: Clasificador MLP para datos en espiral (versión profesional)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 🔹 Paso 1: Función para generar datos en espiral no lineal
def generar_espiral(n=5000, vueltas=3, ruido=0.5):
    """
    Genera dos espirales entrelazadas (problema no lineal) con ruido.

    Args:
        n (int): cantidad de puntos por clase
        vueltas (int): cantidad de vueltas que da cada espiral
        ruido (float): desviación estándar del ruido aleatorio

    Returns:
        X (ndarray): datos de entrada, shape (2*n, 2)
        y (ndarray): etiquetas binarias (0 o 1), shape (2*n,)
    """
    X, y = [], []
    for clase in range(2):
        t = np.linspace(0, vueltas * np.pi, n)
        r = t
        if clase == 1:
            t += np.pi  # gira la segunda espiral
        x1 = r * np.cos(t) + np.random.normal(0, ruido, n)
        x2 = r * np.sin(t) + np.random.normal(0, ruido, n)
        X.append(np.stack((x1, x2), axis=1))
        y.append(np.full(n, clase))
    return np.vstack(X), np.concatenate(y)


# 🔹 Paso 2: Generar y escalar los datos
X, y = generar_espiral(n=50000, vueltas=3, ruido=0.2)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Paso 3: Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Paso 4: Definir y entrenar el modelo MLP
model = MLPClassifier(
    hidden_layer_sizes=(64, 64, 64),  # 3 capas ocultas de 64 neuronas
    activation="relu",
    solver="adam",
    max_iter=5000,
    random_state=42,
    early_stopping=True,  # para evitar sobreajuste
    n_iter_no_change=10,  # paciencia de 10 iteraciones sin mejora
)
model.fit(X_train, y_train)

# 🔹 Paso 5: Evaluación del modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy en test: {acc:.4f}")
print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Clase 0", "Clase 1"]))

print("\n📈 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 🔹 Paso 6: Visualización de las predicciones
plt.figure(figsize=(8, 8))
colores = ["red" if label == 0 else "green" for label in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=10, edgecolors="k", alpha=0.6)
plt.title("Clasificación de datos en espiral con Red Neuronal (MLP)", fontsize=14)
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.tight_layout()
plt.show()
