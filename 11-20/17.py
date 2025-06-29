# Ejercicio 17: Clasificación de puntos en una figura de estrella doble (multi-borde)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def generar_datos_estrella_doble(n=100_000, centro=(0, 0), escala=1.0, puntas=5):
    """
    Genera puntos distribuidos en una figura con forma de estrella doble (dos capas de picos).

    Parámetros:
    - n: número de puntos a generar.
    - centro: coordenadas del centro de la figura (x, y).
    - escala: factor para ampliar o reducir la figura.
    - puntas: número de picos de la estrella.

    Retorna:
    - X: coordenadas de los puntos (n x 2).
    - y: etiquetas binarias (1 = dentro de la estrella, 0 = fuera).
    """
    # Generar puntos aleatorios dentro de un cuadrado centrado en el origen
    X = (np.random.rand(n, 2) * 2 - 1) * escala * 3
    x, y_coord = X[:, 0] - centro[0], X[:, 1] - centro[1]

    # Convertir a coordenadas polares
    r = np.sqrt(x**2 + y_coord**2)
    theta = np.mod(np.arctan2(y_coord, x), 2 * np.pi)

    # Dos radios sinusoidales desfasados: uno grande y uno más pequeño
    r_estrella_externa = 1 + 0.3 * np.cos(puntas * theta)
    r_estrella_interna = 0.7 + 0.3 * np.cos(puntas * theta + np.pi / puntas)

    # Clasificación: dentro de cualquiera de las dos formas
    dentro = (r < r_estrella_externa * escala) | (r < r_estrella_interna * escala)
    y = dentro.astype(int)

    return X, y


# Paso 1: Generar datos
X, y = generar_datos_estrella_doble(n=200_000, centro=(0, 0), escala=2.5, puntas=6)

# Paso 2: Normalización con MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Paso 3: Separar los datos en entrenamiento y prueba, manteniendo la proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Paso 4: Definir y entrenar el modelo de Red Neuronal Multicapa (MLP)
model = MLPClassifier(
    hidden_layer_sizes=(64, 64),  # Dos capas ocultas con 64 neuronas cada una
    activation="relu",  # Activación no lineal ReLU
    solver="adam",  # Optimizador eficiente para redes profundas
    max_iter=3000,  # Mayor número de iteraciones para asegurar convergencia
    random_state=42,
)
model.fit(X_train, y_train)

# Paso 5: Predicciones y evaluación
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Resultados del modelo
print(f"\n✅ Accuracy del modelo: {acc:.4f}")
print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred, target_names=["Fuera de la estrella", "Dentro de la estrella"]
    )
)

print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Paso 6: Visualización de resultados
plt.figure(figsize=(8, 8))
colores = ["red" if pred == 0 else "green" for pred in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=10, edgecolors="k", alpha=0.5)
plt.title("Clasificación de puntos en figura de estrella doble - Red Neuronal")
plt.xlabel("x1 (horizontal)")
plt.ylabel("x2 (vertical)")
plt.grid(True)
plt.tight_layout()
plt.show()
