# Ejercicio 8 (versión profesional)
# Objetivo: Aprender la función lógica XOR usando una red neuronal (MLPClassifier)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")  # Suprime warnings por estética (ej: convergencia)

# -------------------------------------------------
# 1. Datos de entrada: la tabla de verdad XOR
# -------------------------------------------------
# XOR solo da 1 si los valores son diferentes
# Entrada:
# x1 | x2 | XOR
#  0 |  0 |  0
#  0 |  1 |  1
#  1 |  0 |  1
#  1 |  1 |  0

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([0, 1, 1, 0])  # Salida esperada de XOR

# -------------------------------------------------
# 2. Modelo: Red neuronal con sklearn
# -------------------------------------------------
# MLP = Multi-Layer Perceptron (Perceptrón multicapa)
model = MLPClassifier(
    hidden_layer_sizes=(10,),  # 1 capa oculta con 10 neuronas
    activation="relu",  # Función de activación no lineal
    solver="adam",  # Optimizador moderno (basado en gradiente)
    max_iter=3000,  # Iteraciones suficientes para aprender
    random_state=42,  # Reproducibilidad
)

# Entrenamos el modelo con los datos XOR
model.fit(X, y)

# -------------------------------------------------
# 3. Evaluación del modelo
# -------------------------------------------------
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

print(f"\n✅ Accuracy del modelo XOR: {acc:.4f}")

# Reporte de clasificación: métricas por clase
print("\n📋 Reporte de clasificación:")
print(classification_report(y, y_pred, target_names=["Clase 0", "Clase 1"]))

# Matriz de confusión: compara predicciones vs reales
print("\n📊 Matriz de confusión:")
print(confusion_matrix(y, y_pred))

# -------------------------------------------------
# 4. Visualización gráfica de las predicciones
# -------------------------------------------------
# Colores para representar la clase predicha
colores = ["red" if pred == 0 else "green" for pred in y_pred]

plt.figure(figsize=(6, 6))
plt.title("🧠 Clasificación XOR aprendida por Red Neuronal", fontsize=14)

# Muestra cada punto con su color predicho
plt.scatter(X[:, 0], X[:, 1], c=colores, s=200, edgecolor="black")

# Anota la predicción al lado del punto
for i, punto in enumerate(X):
    plt.text(punto[0] + 0.05, punto[1] + 0.05, f"Pred: {y_pred[i]}", fontsize=12)

# Detalles del gráfico
plt.xlabel("x1")
plt.ylabel("x2")
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.grid(True)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()
