# 🧪 Ejercicio 39/200: Clasificación de prendas Fashion MNIST con MLP (Keras)
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

# 🔁 Reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# 1️⃣ Carga del dataset Fashion MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2️⃣ Preprocesamiento de datos
# - Reescalar: pasar a rango [0, 1]
# - Aplanar las imágenes de 28x28 a vectores de 784 para el MLP
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

# - One-hot encoding de las etiquetas
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# 3️⃣ Construcción del modelo MLP
model = Sequential(
    [
        Dense(128, activation="relu", input_shape=(784,), name="CapaOculta1"),
        Dense(64, activation="relu", name="CapaOculta2"),
        Dense(32, activation="relu", name="CapaOculta3"),
        Dense(10, activation="softmax", name="SalidaSoftmax"),
    ]
)

# 4️⃣ Compilación del modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 5️⃣ Entrenamiento del modelo
history = model.fit(
    X_train, y_train_cat, validation_split=0.1, epochs=20, batch_size=128, verbose=1
)

# 6️⃣ Evaluación del modelo en el conjunto de prueba
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"✅ Precisión en test: {acc:.4f}")

# 7️⃣ Visualización de curvas de entrenamiento
plt.figure(figsize=(12, 5))

# Curva de pérdida (entrenamiento vs validación)
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.title("Evolución de la Pérdida")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Curva de precisión (entrenamiento vs validación)
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Evolución de la Precisión")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
