# Ejercicio 7 Profesional: Clasificador con red neuronal (MLPClassifier)
# Objetivo: Aprender a clasificar si un punto (x, y) está dentro o fuera de un círculo.

import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")  # Opcional: suprime warnings innecesarios

# -------------------------------------------------
# 1. Generación del dataset sintético
# -------------------------------------------------

# Creamos 100.000 puntos aleatorios dentro de un cuadrado de 10x10
X = np.random.rand(100000, 2) * 10

# Definimos un círculo de centro (5,5) y radio 3
cx, cy = 5, 5
r = 3

# Calculamos la distancia cuadrada desde cada punto al centro del círculo
distancia_cuadrada = (X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2

# Asignamos etiqueta 1 si está dentro del círculo, 0 si está fuera
y = (distancia_cuadrada < r**2).astype(int)

# -------------------------------------------------
# 2. Preprocesamiento: Escalado de características
# -------------------------------------------------

# Normalizamos los datos entre 0 y 1 (obligatorio para redes neuronales)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# 3. División del dataset
# -------------------------------------------------

# Separamos datos en entrenamiento y prueba (80/20)
# Usamos stratify para mantener el equilibrio entre clases
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# 4. Entrenamiento de la Red Neuronal
# -------------------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(30, 30),  # 2 capas ocultas con 30 neuronas cada una
    activation="relu",  # Función de activación no lineal
    solver="adam",  # Optimizador moderno basado en gradiente
    max_iter=1000,  # Iteraciones para asegurar convergencia
    random_state=42,
)

# Entrenamos el modelo con los datos normalizados
model.fit(X_train, y_train)

# -------------------------------------------------
# 5. Evaluación del modelo
# -------------------------------------------------

# Predicción sobre los datos de prueba
y_pred = model.predict(X_test)

# Métricas de rendimiento
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy del modelo (Red Neuronal): {acc:.4f}")

print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred, target_names=["Fuera del círculo", "Dentro del círculo"]
    )
)

print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------
# 6. Predicciones manuales
# -------------------------------------------------

print("\n🔍 Predicciones para puntos manuales:")
ejemplos = [(5, 5), (2, 2), (8, 5), (5, 2), (6, 8)]

for punto in ejemplos:
    punto_scaled = scaler.transform([punto])  # Usar transform, no fit_transform
    pred = model.predict(punto_scaled)[0]
    etiqueta = "Dentro del círculo" if pred == 1 else "Fuera del círculo"
    print(f"Punto {punto} → Clase predicha: {pred} ({etiqueta})")

# -------------------------------------------------
# 7. Visualización: Datos + Círculo + Puntos manuales
# -------------------------------------------------

# Para graficar, desnormalizamos los datos a su escala original
X_test_original = scaler.inverse_transform(X_test)

# Colores por clase: azul = fuera, rojo = dentro
plt.figure(figsize=(8, 8))
plt.scatter(
    X_test_original[:, 0],
    X_test_original[:, 1],
    c=y_pred,
    cmap="coolwarm",
    alpha=0.4,
    edgecolors="none",
    s=10,
    label="Predicción del modelo",
)

# Dibuja el círculo real (para comparar con la predicción)
circle = plt.Circle((cx, cy), r, color="black", fill=False, linestyle="--", linewidth=2)
plt.gca().add_patch(circle)

# Puntos de prueba manual
for punto in ejemplos:
    plt.plot(punto[0], punto[1], "ko", markersize=8, label="Punto manual")

plt.title(
    "🔵 Clasificación de puntos por red neuronal\n(Círculo centro (5,5), radio=3)"
)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis("equal")
plt.legend(["Círculo real", "Puntos manuales"])
plt.show()
