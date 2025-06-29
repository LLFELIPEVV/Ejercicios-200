# 🧪 Ejercicio 19: Clasificación de puntos dentro de una curva de mariposa usando una Red Neuronal MLP

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def generar_datos_mariposa(n=100_000, escala=1.0):
    """
    Genera puntos aleatorios y clasifica si caen dentro de la curva polar conocida como "mariposa".

    Parámetros:
    - n: cantidad de puntos a generar
    - escala: factor para escalar la figura
    Retorna:
    - X: coordenadas de los puntos (n x 2)
    - y: etiquetas binarias (1 si está dentro de la mariposa, 0 si está fuera)
    """

    # Generamos puntos en el rango [-3, 3] para ambos ejes
    X = (np.random.rand(n, 2) * 2 - 1) * escala * 3
    x, y = X[:, 0], X[:, 1]

    # Convertimos a coordenadas polares
    r = np.sqrt(x**2 + y**2)
    theta = np.mod(np.arctan2(y, x), 2 * np.pi)

    # Curva mariposa (fórmula modificada para facilitar clasificación)
    r_mariposa = (
        np.exp(np.cos(theta)) - 2 * np.cos(4 * theta) + (np.sin(theta / 12) ** 5)
    )

    # Etiquetamos: 1 si el punto está dentro de la curva, 0 si está fuera
    dentro = r < r_mariposa * escala
    y_labels = dentro.astype(int)

    return X, y_labels


# 🔹 Paso 1: Generar los datos
X, y = generar_datos_mariposa(n=200_000, escala=2.5)

# 🔹 Paso 2: Normalización para escalar valores a rango [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Paso 3: Separar datos en entrenamiento y testeo (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 🔹 Paso 4: Definir y entrenar el modelo MLP
model = MLPClassifier(
    hidden_layer_sizes=(64, 64),  # Dos capas ocultas con 64 neuronas cada una
    activation="relu",  # Función de activación ReLU
    solver="adam",  # Optimizador Adam
    max_iter=3000,  # Número máximo de iteraciones
    random_state=42,  # Semilla para reproducibilidad
)

model.fit(X_train, y_train)

# 🔹 Paso 5: Evaluación del modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy del modelo: {acc:.4f}")
print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred, target_names=["Fuera de la mariposa", "Dentro de la mariposa"]
    )
)

print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 🔹 Paso 6: Visualización de las predicciones
plt.figure(figsize=(8, 8))
colores = ["red" if pred == 0 else "green" for pred in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=10, edgecolors="k", alpha=0.5)
plt.title("Clasificación de puntos - Curva de Mariposa", fontsize=14)
plt.xlabel("x1 (horizontal)")
plt.ylabel("x2 (vertical)")
plt.grid(True)
plt.tight_layout()
plt.show()
