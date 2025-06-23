# Ejercicio 11: Clasificación con Árbol de Decisión en datos no lineales (regiones cuadradas)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 🔧 Función para generar datos con patrones en regiones cuadradas
def generar_datos_regiones(n=10000):
    """
    Genera datos aleatorios (x1, x2) con etiquetas según si están dentro
    de dos regiones cuadradas definidas:
    - Región 1: x1 < 3 y x2 < 3
    - Región 2: x1 > 7 y x2 > 7
    Las demás zonas son etiquetadas como clase 0.
    """
    X = np.random.rand(n, 2) * 10  # Datos en rango [0, 10)
    x1, x2 = X[:, 0], X[:, 1]

    en_region1 = (x1 < 3) & (x2 < 3)
    en_region2 = (x1 > 7) & (x2 > 7)
    y = (en_region1 | en_region2).astype(int)
    return X, y


# 🧪 Generación del dataset
X, y = generar_datos_regiones(100000)

# ✂️ Dividimos en entrenamiento y prueba, preservando proporciones de clases
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # mantiene proporción de clases en train y test
)

# 🌳 Modelo: Árbol de Decisión limitado a 4 niveles para evitar sobreajuste
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 🎯 Evaluación
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc:.4f}")
print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Fuera", "Dentro"]))

print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 📈 Importancia de cada variable (x1 y x2)
print("\n📌 Importancia de las características (feature importance):")
for i, score in enumerate(model.feature_importances_):
    print(f"Feature x{i + 1}: {score:.3f}")

# 🌳 Visualización del árbol aprendido
plt.figure(figsize=(12, 6))
plot_tree(
    model,
    feature_names=["x1", "x2"],
    class_names=["Clase 0 (Fuera)", "Clase 1 (Dentro)"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Árbol de Decisión - Clasificación por regiones cuadradas")
plt.show()

# 🧭 Visualización de las predicciones sobre los puntos reales
plt.figure(figsize=(7, 7))
colores = ["red" if pred == 0 else "green" for pred in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=20, edgecolors="k", alpha=0.6)
plt.title("Clasificación de puntos según región (Árbol de Decisión)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.tight_layout()
plt.show()
