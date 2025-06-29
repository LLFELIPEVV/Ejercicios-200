# 🧪 Ejercicio 20: Clasificación de espirales triples usando una red neuronal MLP

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def generar_espirales_multiclase(n_por_clase=2000, clases=3, ruido=0.2):
    """
    Genera datos sintéticos en forma de múltiples espirales entrelazadas.

    Parámetros:
    - n_por_clase (int): número de puntos por clase
    - clases (int): número de espirales (clases)
    - ruido (float): desviación estándar del ruido añadido

    Retorna:
    - X (np.ndarray): matriz de características (n_samples, 2)
    - y (np.ndarray): vector de etiquetas (n_samples,)
    """
    X, y = [], []

    for c in range(clases):
        t = np.linspace(0, 4 * np.pi, n_por_clase)
        r = t  # radio crece con t, creando una espiral
        offset = (2 * np.pi * c) / clases  # desfase angular para separar espirales

        # Coordenadas cartesianas + ruido gaussiano
        x1 = r * np.cos(t + offset) + np.random.normal(0, ruido, n_por_clase)
        x2 = r * np.sin(t + offset) + np.random.normal(0, ruido, n_por_clase)

        X.append(np.column_stack((x1, x2)))
        y.append(np.full(n_por_clase, c))  # etiqueta de la clase c

    return np.vstack(X), np.concatenate(y)


# 🔹 Paso 1: Generar los datos de espirales
X, y = generar_espirales_multiclase(n_por_clase=20_000, clases=3, ruido=0.25)

# 🔹 Paso 2: Escalar características al rango [0, 1] para estabilizar el entrenamiento
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Paso 3: División en entrenamiento y prueba (stratify mantiene proporciones de clase)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Paso 4: Crear y entrenar el modelo MLP
model = MLPClassifier(
    hidden_layer_sizes=(64, 64),  # dos capas ocultas con 64 neuronas cada una
    activation="relu",  # función de activación ReLU
    solver="adam",  # optimizador Adam
    max_iter=3000,  # más iteraciones para asegurar convergencia
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
        y_test, y_pred, target_names=["Clase 0", "Clase 1", "Clase 2"]
    )
)

print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 🔹 Paso 6: Visualización de las predicciones
plt.figure(figsize=(8, 8))
colores = np.array(["red", "green", "blue"])  # Colores por clase
plt.scatter(
    X_test[:, 0], X_test[:, 1], c=colores[y_pred], s=10, edgecolors="k", alpha=0.6
)
plt.title("Clasificación de espirales múltiples - Red Neuronal MLP", fontsize=14)
plt.xlabel("x1 (horizontal)")
plt.ylabel("x2 (vertical)")
plt.grid(True)
plt.tight_layout()
plt.show()
