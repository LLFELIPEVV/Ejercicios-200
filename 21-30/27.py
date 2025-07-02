# 🧪 Visualización de la frontera de decisión usando MLP con el dataset Iris (2 características)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

# 1️⃣ Cargar el dataset Iris y usar solo dos características para poder graficar en 2D
iris = load_iris()
X = iris.data[:, :2]  # Solo las primeras 2 características: sépalo largo y ancho
y = iris.target
nombres_clases = iris.target_names

# 2️⃣ Normalizar los datos para que todas las características estén en la misma escala
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Dividir los datos en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Definir una grilla amplia de hiperparámetros para buscar la mejor configuración del MLP
param_grid = {
    "hidden_layer_sizes": [
        (2,),
        (4,),
        (4, 2),
        (8,),
        (8, 4),
        (8, 4, 2),
        (16,),
        (16, 8),
        (16, 8, 4),
        (16, 8, 4, 2),
        (32,),
        (32, 16),
        (32, 16, 8),
        (32, 16, 8, 4),
        (32, 16, 8, 4, 2),
        (64,),
        (64, 32),
        (64, 32, 16),
        (64, 32, 16, 8),
        (64, 32, 16, 8, 4),
        (64, 32, 16, 8, 4, 2),
    ],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2],  # Parámetro de regularización L2
    "activation": ["relu", "tanh"],  # Funciones de activación comunes
}

# 5️⃣ Instanciar el modelo base de MLP (optimizador Adam, sin definir arquitectura aún)
mlp = MLPClassifier(solver="adam", max_iter=3000, random_state=42)

# 6️⃣ Realizar búsqueda en grilla con validación cruzada de 3 folds
grid = GridSearchCV(
    estimator=mlp, param_grid=param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=2
)
grid.fit(X_train, y_train)

# 7️⃣ Mostrar el mejor modelo encontrado
print(f"\n🏆 Mejor accuracy validación cruzada: {grid.best_score_:.4f}")
print("🔧 Mejores hiperparámetros encontrados:")
for param, value in grid.best_params_.items():
    print(f"   {param}: {value}")

# 8️⃣ Evaluar el modelo óptimo sobre el conjunto de prueba
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# 9️⃣ Generar una malla (rejilla) de puntos sobre todo el espacio 2D para visualizar la frontera
h = 0.001  # paso muy fino para mayor precisión visual
x_min, x_max = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
y_min, y_max = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
puntos_malla = np.c_[
    xx.ravel(), yy.ravel()
]  # Convertir malla a array de coordenadas 2D

# 🔟 Predecir clase para cada punto de la malla
Z = best_model.predict(puntos_malla)
Z = Z.reshape(xx.shape)  # Volver a forma de matriz para graficar

# 🔁 Visualización de la frontera de decisión y datos reales
plt.figure(figsize=(8, 6))
plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    levels=[-0.5, 0.5, 1.5, 2.5],
    colors=["#FFAAAA", "#AAFFAA", "#AAAAFF"],
)

# 🔹 Puntos de entrenamiento y prueba con etiquetas de clase
plt.scatter(
    X_train[:, 0], X_train[:, 1], c=y_train, cmap="jet", edgecolor="k", label="Train"
)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="jet", marker="x", label="Test")

# Etiquetas del gráfico
plt.xlabel("Sépalo largo (normalizado)")
plt.ylabel("Sépalo ancho (normalizado)")
plt.title("🌼 Frontera de decisión de Red Neuronal MLP - Dataset Iris")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
