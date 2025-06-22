# Ejercicio 10: Clasificador no lineal (MLP) para identificar puntos dentro de un anillo
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ----------------------------
# 🌀 FUNCIONES AUXILIARES
# ----------------------------
def generar_donut(n=10000, radio_interno=2.5, radio_externo=4.5):
    """
    Genera un conjunto de puntos aleatorios en un plano 2D y los clasifica según si están
    dentro de un anillo circular (donut) definido por radio_interno y radio_externo.

    Parámetros:
        - n: cantidad de puntos a generar
        - radio_interno: radio menor del anillo
        - radio_externo: radio mayor del anillo

    Retorna:
        - X: array (n, 2) de coordenadas
        - y: array (n,) de etiquetas (1 si está dentro del anillo, 0 si no)
    """
    X = np.random.rand(n, 2) * 10  # puntos aleatorios entre (0,0) y (10,10)
    centro_x, centro_y = 5, 5  # centro del anillo
    distancias = np.sqrt((X[:, 0] - centro_x) ** 2 + (X[:, 1] - centro_y) ** 2)
    y = ((distancias > radio_interno) & (distancias < radio_externo)).astype(int)
    return X, y


# ----------------------------
# 📊 GENERACIÓN DE DATOS
# ----------------------------
X, y = generar_donut(n=2000)  # Datos para clasificación binaria no lineal

# Dividimos en entrenamiento y prueba (estratificando para mantener proporción de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# 🤖 ENTRENAMIENTO DEL MODELO
# ----------------------------

# Red neuronal con 2 capas ocultas de 30 neuronas cada una
model = MLPClassifier(
    hidden_layer_sizes=(30, 30),
    activation="relu",  # Función de activación no lineal
    solver="adam",  # Optimizador moderno y eficiente
    max_iter=2000,
    random_state=42,
)

model.fit(X_train, y_train)

# ----------------------------
# 🧠 EVALUACIÓN DEL MODELO
# ----------------------------

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy en test: {acc:.4f}")
print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred, target_names=["Fuera del anillo", "Dentro del anillo"]
    )
)

print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# 📈 VISUALIZACIÓN DE RESULTADOS
# ----------------------------

plt.figure(figsize=(8, 8))
colores = ["red" if pred == 0 else "green" for pred in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=30, edgecolors="k", alpha=0.6)
plt.title("🔍 Clasificación de puntos dentro de un anillo (MLPClassifier)", fontsize=14)
plt.xlabel("x1")
plt.ylabel("x2")

# Dibujamos el anillo real como referencia
circulo_interno = plt.Circle(
    (5, 5), 2.5, color="blue", fill=False, linestyle="--", label="Radio interno"
)
circulo_externo = plt.Circle(
    (5, 5), 4.5, color="purple", fill=False, linestyle="--", label="Radio externo"
)
plt.gca().add_patch(circulo_interno)
plt.gca().add_patch(circulo_externo)

plt.legend(loc="upper right")
plt.grid(True)
plt.axis("equal")
plt.show()
