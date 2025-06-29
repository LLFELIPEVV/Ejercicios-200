# 🧪 Ejercicio 15: Clasificación de puntos dentro de una estrella usando una red neuronal MLP

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def generar_datos_estrella(n=100000, centro=(0, 0), escala=1.0, puntas=5):
    """
    Genera puntos (x, y) aleatorios y clasifica si caen dentro de una estrella polar.

    Parámetros:
    - n: número de muestras
    - centro: centro de la estrella (por defecto en el origen)
    - escala: tamaño de la figura
    - puntas: número de puntas de la estrella (por defecto 5)

    Retorna:
    - X: matriz (n x 2) de coordenadas
    - y: vector de etiquetas binario (1 = dentro, 0 = fuera)
    """
    # Generamos puntos aleatorios en un cuadrado centrado en el origen
    X = (np.random.rand(n, 2) * 2 - 1) * escala * 3

    # Centramos los puntos respecto al centro dado
    x, y_coord = X[:, 0] - centro[0], X[:, 1] - centro[1]

    # Convertimos a coordenadas polares
    r = np.sqrt(x**2 + y_coord**2)  # radio
    theta = np.arctan2(y_coord, x)  # ángulo
    theta = np.mod(theta, 2 * np.pi)  # convertimos a rango [0, 2π]

    # Fórmula polar para estrella con 'puntas' lóbulos
    r_estrella = 1 + 0.5 * np.cos(puntas * theta)

    # Clasificación: dentro si r es menor al radio de la estrella en ese ángulo
    dentro = r < r_estrella * escala
    etiquetas = dentro.astype(int)

    return X, etiquetas


# 🔹 Paso 1: Generar los datos
X, y = generar_datos_estrella(
    n=100000,  # muestra amplia para buen entrenamiento
    centro=(0, 0),
    escala=2.5,
    puntas=5,
)

# 🔹 Paso 2: Escalar los datos (rango [0, 1])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Paso 3: Dividir en entrenamiento y prueba (con estratificación para balancear clases)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Paso 4: Definir y entrenar la red neuronal
model = MLPClassifier(
    hidden_layer_sizes=(64, 64),  # arquitectura con dos capas ocultas de 64 neuronas
    activation="relu",
    solver="adam",
    max_iter=3000,
    random_state=42,
)
model.fit(X_train, y_train)

# 🔹 Paso 5: Evaluación del modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy del modelo: {acc:.4f}")
print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred, target_names=["Fuera de la estrella", "Dentro de la estrella"]
    )
)
print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 🔹 Paso 6: Visualización de la clasificación
plt.figure(figsize=(8, 8))
colores = ["red" if p == 0 else "green" for p in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=10, edgecolors="k", alpha=0.5)
plt.title("Clasificación de puntos en forma de estrella - Red Neuronal")
plt.xlabel("x1 (horizontal)")
plt.ylabel("x2 (vertical)")
plt.grid(True)
plt.tight_layout()
plt.show()
