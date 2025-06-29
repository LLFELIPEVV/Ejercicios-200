# Ejercicio 12: Clasificación de datos no lineales (media luna) usando una red neuronal MLP
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 📌 Paso 1: Generación de datos sintéticos con forma de media luna
# - `noise`: añade ruido para que el problema no sea perfectamente lineal
# - `random_state`: garantiza reproducibilidad
X, y = make_moons(n_samples=100_000, noise=0.25, random_state=42)

# 📌 Paso 2: Escalamiento de características
# - Escalamos los datos entre 0 y 1 para mejorar la eficiencia del entrenamiento
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Paso 3: División en conjuntos de entrenamiento y prueba
# - `stratify=y` asegura que la proporción de clases sea la misma en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 📌 Paso 4: Definición del modelo de red neuronal (MLPClassifier)
# - 2 capas ocultas con 20 neuronas cada una
# - Función de activación ReLU para no linealidad
# - Optimizador Adam, eficiente para muchos problemas
model = MLPClassifier(
    hidden_layer_sizes=(20, 20),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42,
)

# 📌 Paso 5: Entrenamiento del modelo
model.fit(X_train, y_train)

# 📌 Paso 6: Evaluación del modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy en test: {acc:.4f}")
print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Clase 0", "Clase 1"]))
print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 📌 Paso 7: Visualización de las predicciones sobre el conjunto de prueba
plt.figure(figsize=(7, 7))
colores = ["red" if p == 0 else "green" for p in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colores, s=20, edgecolors="k", alpha=0.6)
plt.title("Clasificación de datos en forma de media luna (MLPClassifier)")
plt.xlabel("x1 (característica 1)")
plt.ylabel("x2 (característica 2)")
plt.grid(True)
plt.tight_layout()
plt.show()
