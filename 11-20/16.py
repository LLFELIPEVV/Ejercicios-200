# Ejercicio 16: 🌸 Clasificación de puntos dentro de una flor polar usando una Red Neuronal MLP

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def generar_datos_flor_polar(n=100_000, centro=(0, 0), escala=1.0, petalos=6):
    """
    Genera puntos distribuidos aleatoriamente y clasifica si están dentro de una flor polar.
    Fórmula de la flor: r(θ) = 1 + 0.5 * cos(kθ), donde k = número de pétalos.

    Parámetros:
        n (int): Número de puntos a generar.
        centro (tuple): Coordenadas del centro de la flor (x, y).
        escala (float): Factor de escala para el tamaño de la flor.
        petalos (int): Número de pétalos que tendrá la flor.

    Retorna:
        X (ndarray): Coordenadas (x1, x2) de cada punto.
        y (ndarray): Etiquetas binarias (1 si está dentro de la flor, 0 si está fuera).
    """
    if petalos < 1 or n < 1:
        raise ValueError("El número de puntos y de pétalos debe ser positivo.")

    # Generar puntos aleatorios en un cuadrado centrado en el origen
    X = (np.random.rand(n, 2) * 2 - 1) * escala * 3
    x, y_coord = X[:, 0] - centro[0], X[:, 1] - centro[1]

    # Convertir a coordenadas polares
    r = np.sqrt(x**2 + y_coord**2)
    theta = np.arctan2(y_coord, x)
    theta = np.mod(theta, 2 * np.pi)

    # Definir radio de la flor en función de theta
    radio_flor = 1 + 0.5 * np.cos(petalos * theta)

    # Etiqueta 1 si el punto está dentro del contorno de la flor
    esta_dentro = r < radio_flor * escala
    return X, esta_dentro.astype(int)


# 🔹 Paso 1: Generar los datos
X, y = generar_datos_flor_polar(n=200_000, centro=(0, 0), escala=2.5, petalos=6)

# 🔹 Paso 2: Escalar características entre 0 y 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Paso 3: Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 🔹 Paso 4: Construcción del modelo (Red Neuronal Multicapa)
model = MLPClassifier(
    hidden_layer_sizes=(64, 64),
    activation="relu",  # Activación no lineal para aprender patrones complejos
    solver="adam",  # Optimizador eficiente y ampliamente utilizado
    max_iter=3000,  # Iteraciones suficientes para converger
    random_state=42,
)

# 🔹 Paso 5: Entrenar el modelo
model.fit(X_train, y_train)

# 🔹 Paso 6: Evaluar el desempeño del modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy del modelo: {acc:.4f}")

print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred, target_names=["Fuera de la flor", "Dentro de la flor"]
    )
)

print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 🔹 Paso 7: Visualización gráfica de la clasificación
plt.figure(figsize=(8, 8))
colores = ["red" if pred == 0 else "green" for pred in y_pred]

plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=10, edgecolors="k", alpha=0.5)

plt.title("Clasificación de puntos dentro de una flor polar - Red Neuronal")
plt.xlabel("x1 (horizontal)")
plt.ylabel("x2 (vertical)")
plt.grid(True)
plt.tight_layout()
plt.show()
