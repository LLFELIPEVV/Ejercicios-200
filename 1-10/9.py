# Ejercicio 9: Clasificador XOR con ruido usando red neuronal (MLP)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ----------------------------
# 🔧 Generador de datos XOR con ruido
# ----------------------------
def generar_datos_xor(centro, clase, ruido=0.2, n=1000):
    """
    Genera n puntos aleatorios cercanos al 'centro' con distribución normal
    y les asigna la misma clase.
    """
    X = np.random.normal(loc=centro, scale=ruido, size=(n, 2))
    y = np.full(n, clase)
    return X, y


# ----------------------------
# 🎯 Definimos los centros y sus clases XOR
# ----------------------------
# Puntos base de la compuerta XOR con sus respectivas clases
# (0 XOR 0 = 0), (0 XOR 1 = 1), (1 XOR 0 = 1), (1 XOR 1 = 0)
centros_clases = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

# Generar datos para cada centro
X_partes, y_partes = [], []
for centro, clase in centros_clases:
    X_centro, y_centro = generar_datos_xor(centro, clase, ruido=0.2, n=1000)
    X_partes.append(X_centro)
    y_partes.append(y_centro)

# Concatenar todos los puntos y etiquetas en un solo dataset
X = np.vstack(X_partes)
y = np.concatenate(y_partes)

# ----------------------------
# ✂️ Separar datos en entrenamiento y prueba
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# 🧠 Definir y entrenar el modelo de red neuronal
# ----------------------------
model = MLPClassifier(
    hidden_layer_sizes=(20, 20),  # Dos capas ocultas con 20 neuronas cada una
    activation="relu",  # Función de activación no lineal
    solver="adam",  # Optimizador moderno y eficiente
    max_iter=2000,  # Iteraciones suficientes para convergencia
    random_state=42,
)

model.fit(X_train, y_train)

# ----------------------------
# 📈 Evaluación del modelo
# ----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy en test (XOR con ruido): {acc:.4f}\n")
print("📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Clase 0", "Clase 1"]))

print("📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# 📊 Visualización de resultados
# ----------------------------
plt.figure(figsize=(7, 7))
plt.title("Clasificación de patrón XOR con ruido", fontsize=14)

# Colores de predicción: verde si predijo clase 1, rojo si predijo clase 0
colores = ["green" if label == 1 else "red" for label in y_pred]

# Graficamos los puntos de prueba con colores según la predicción
plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=30, alpha=0.6, edgecolors="black")

# Añadir etiquetas y cuadrícula
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.legend(
    handles=[
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Clase 0 (Rojo)",
            markerfacecolor="red",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Clase 1 (Verde)",
            markerfacecolor="green",
            markersize=10,
        ),
    ]
)
plt.tight_layout()
plt.show()
