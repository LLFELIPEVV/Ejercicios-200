# 🧠 Ejercicio 21: Clasificación de dígitos escritos a mano con Red Neuronal MLP

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Cargar el dataset de dígitos (cada imagen es de 8x8 píxeles, aplanada en un vector de 64 dimensiones)
digits = load_digits()
X = digits.data  # Datos de entrada (imagen aplanada)
y = digits.target  # Etiquetas (dígitos del 0 al 9)

# 2️⃣ Escalar los valores de píxeles a [0, 1] — importante para redes neuronales
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ División en conjuntos de entrenamiento y prueba (estratificada para mantener proporciones de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 4️⃣ Definición del modelo MLP (Multi-Layer Perceptron)
# - 1 capa oculta con 100 neuronas
# - ReLU como función de activación
# - Optimizador Adam
# - Hasta 3000 iteraciones para asegurar convergencia
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    max_iter=3000,
    random_state=42,
)

# 5️⃣ Entrenamiento del modelo
model.fit(X_train, y_train)

# 6️⃣ Predicción sobre los datos de prueba
y_pred = model.predict(X_test)

# 7️⃣ Evaluación del modelo
acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred)

# 8️⃣ Reporte en consola
print(f"\n✅ Accuracy del modelo en test: {acc:.4f}")
print("\n📋 Reporte de clasificación por clase:")
print(rep)
print("\n📊 Matriz de confusión (errores por clase):")
print(conf)

# 9️⃣ Visualización de predicciones sobre imágenes
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Predicciones del modelo sobre imágenes reales (test)", fontsize=14)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
    ax.set_title(f"Pred: {y_pred[i]}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()
