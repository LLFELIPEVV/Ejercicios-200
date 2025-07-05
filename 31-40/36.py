# 🧠 Ejercicio 36: Clasificación del Iris usando un MLP con Keras

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# 1️⃣ Cargar datos Iris
iris = load_iris()
X, y = iris.data, iris.target
n_clases = len(np.unique(y))  # 3 clases: setosa, versicolor, virginica

# 2️⃣ Escalar características (normalización)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Codificación one-hot de las etiquetas
y_encoded = to_categorical(y, num_classes=n_clases)

# 4️⃣ Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ Definición del modelo secuencial MLP
model = Sequential(
    [
        Dense(16, input_shape=(X.shape[1],), activation="relu"),  # Capa oculta 1
        Dense(8, activation="relu"),  # Capa oculta 2
        Dense(4, activation="relu"),  # Capa oculta 3
        Dense(2, activation="relu"),  # Capa oculta 4
        Dense(n_clases, activation="softmax"),  # Capa de salida (multiclase)
    ]
)

# 6️⃣ Compilación del modelo
model.compile(
    optimizer="adam",  # Optimización eficiente
    loss="categorical_crossentropy",  # Pérdida para clasificación multiclase
    metrics=["accuracy"],  # Métrica de desempeño principal
)

# 7️⃣ Entrenamiento del modelo
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,  # Parte del entrenamiento usada para validación
    verbose=0,  # No mostrar logs en consola
)

# 8️⃣ Evaluación en datos de prueba
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Precisión en conjunto de prueba: {acc:.4f}")

# 9️⃣ Predicción y decodificación de etiquetas
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Etiqueta predicha
y_true = np.argmax(y_test, axis=1)  # Etiqueta real

# 🔟 Reporte de clasificación
print("\n📋 Reporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=iris.target_names))

# 🔢 Matriz de confusión
conf = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.title("Matriz de Confusión - Clasificación Iris")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Real")
plt.tight_layout()
plt.show()
