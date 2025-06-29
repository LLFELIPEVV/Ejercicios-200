# 🧪 Ejercicio 18: Clasificación de puntos dentro de una figura en forma de ∞ (lazo lemniscata)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def generar_datos_lazo_lemniscata(n=100_000, centro=(0, 0), escala=1.0):
    """
    Genera datos con una forma de lazo ∞ (lemniscata de Bernoulli).
    Los puntos son etiquetados según si están dentro o fuera del contorno.

    Fórmula polar: r² = a² * cos(2θ)
    """
    # Genera puntos aleatorios en el plano [-3, 3] × [-3, 3]
    X = (np.random.rand(n, 2) * 2 - 1) * escala * 3

    # Centrar coordenadas respecto al centro de la figura
    x = X[:, 0] - centro[0]
    y = X[:, 1] - centro[1]

    # Convertir coordenadas cartesianas a polares
    r = np.sqrt(x**2 + y**2)  # Radio
    theta = np.mod(np.arctan2(y, x), 2 * np.pi)  # Ángulo (0 a 2π)

    # Radio del borde de la figura ∞ para ese ángulo
    r_figura = np.sqrt(np.abs(np.cos(2 * theta)))

    # Etiquetar como 1 si está dentro del contorno ∞, 0 si está fuera
    dentro = r < r_figura * escala
    etiquetas = dentro.astype(int)

    return X, etiquetas


# Paso 1️⃣: Generar los datos
X, y = generar_datos_lazo_lemniscata(n=200_000, centro=(0, 0), escala=2.5)

# Paso 2️⃣: Normalizar los datos con MinMaxScaler para que estén entre 0 y 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Paso 3️⃣: Dividir en conjunto de entrenamiento y prueba (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Paso 4️⃣: Definir y entrenar una red neuronal multicapa (MLP)
model = MLPClassifier(
    hidden_layer_sizes=(64, 64),  # Dos capas ocultas con 64 neuronas cada una
    activation="relu",  # Función de activación no lineal
    solver="adam",  # Optimizador adaptativo eficiente
    max_iter=3000,  # Iteraciones máximas para asegurar convergencia
    random_state=42,
)
model.fit(X_train, y_train)

# Paso 5️⃣: Evaluar el modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy del modelo: {acc:.4f}")
print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred, target_names=["Fuera del lazo", "Dentro del lazo"]
    )
)

print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Paso 6️⃣: Visualizar resultados de clasificación
plt.figure(figsize=(8, 8))
colores = ["red" if pred == 0 else "green" for pred in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=10, edgecolors="k", alpha=0.5)
plt.title("Clasificación de puntos en la figura ∞ con Red Neuronal")
plt.xlabel("x1 (horizontal)")
plt.ylabel("x2 (vertical)")
plt.grid(True)
plt.tight_layout()
plt.show()
