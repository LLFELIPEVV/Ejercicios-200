# Ejercicio 14: Clasificación de puntos dentro o fuera de una figura de corazón usando una red neuronal MLP

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def generar_datos_forma_corazon(n_muestras=100000, centro=(0, 0), escala=1.0):
    """
    Genera puntos 2D aleatorios y los etiqueta según si están dentro de una forma de corazón.

    Parámetros:
    - n_muestras: cantidad de puntos a generar.
    - centro: coordenada (x, y) del centro del corazón.
    - escala: factor de escala para aumentar o reducir el tamaño del corazón.

    Retorna:
    - X: array (n_muestras, 2) con coordenadas x1 y x2.
    - y: array (n_muestras,) con etiquetas binarias (0 = fuera, 1 = dentro).
    """
    # Generamos puntos en un rango [-2, 2] y luego los escalamos
    X = (np.random.rand(n_muestras, 2) * 4 - 2) * escala

    # Trasladamos los puntos al centro del corazón
    x = X[:, 0] - centro[0]
    y = X[:, 1] - centro[1]

    # Usamos la fórmula del corazón clásico para determinar si está dentro
    dentro = ((x**2 + y**2 - 1) ** 3 - x**2 * y**3) < 0
    etiquetas = dentro.astype(int)

    return X, etiquetas


# 🔹 Paso 1: Generar los datos
X, y = generar_datos_forma_corazon(n_muestras=20000, centro=(0, 0), escala=1.5)

# 🔹 Paso 2: Escalado (normalizamos para mejorar el rendimiento del modelo)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Paso 3: División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # Asegura misma proporción de clases en ambos subconjuntos
)

# 🔹 Paso 4: Definir y entrenar el modelo MLP (red neuronal multicapa)
model = MLPClassifier(
    hidden_layer_sizes=(64, 64),  # Dos capas ocultas de 64 neuronas cada una
    activation="relu",  # Función de activación no lineal
    solver="adam",  # Optimizador eficiente para grandes datasets
    max_iter=3000,  # Iteraciones máximas para el entrenamiento
    random_state=42,  # Reproducibilidad
)
model.fit(X_train, y_train)

# 🔹 Paso 5: Evaluación del modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy del modelo: {accuracy:.4f}")

print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred, target_names=["Fuera del corazón", "Dentro del corazón"]
    )
)

print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 🔹 Paso 6: Visualización de los resultados
plt.figure(figsize=(8, 8))
colores = ["red" if label == 0 else "green" for label in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=10, edgecolors="k", alpha=0.5)

plt.title("Clasificación de puntos en forma de corazón - Red Neuronal MLP", fontsize=14)
plt.xlabel("x1 (horizontal)")
plt.ylabel("x2 (vertical)")
plt.grid(True)
plt.tight_layout()
plt.show()
