# 🧠 Ejercicio 38: Clasificación de dígitos MNIST con MLP en Keras (versión mejorada)
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# 1️⃣ Cargar el dataset MNIST (60.000 entrenamiento, 10.000 prueba)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2️⃣ Aplanar imágenes 28x28 → 784 y convertir a float32
X_train = X_train.reshape(-1, 784).astype("float32")
X_test = X_test.reshape(-1, 784).astype("float32")

# 3️⃣ Normalizar los píxeles al rango [0, 1]
X_train /= 255.0
X_test /= 255.0

# 4️⃣ One-hot encoding de las etiquetas (0–9)
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)

# 5️⃣ Construir el modelo MLP con 3 capas ocultas
model = Sequential(
    [
        Dense(128, activation="relu", input_shape=(784,)),  # capa de entrada
        Dense(64, activation="relu"),  # capa oculta 1
        Dense(32, activation="relu"),  # capa oculta 2
        Dense(10, activation="softmax"),  # salida softmax para 10 clases
    ]
)

# 6️⃣ Compilar el modelo con función de pérdida adecuada y optimizador Adam
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 7️⃣ Entrenar el modelo con validación del 10%
history = model.fit(
    X_train, y_train_onehot, validation_split=0.1, epochs=20, batch_size=128, verbose=1
)

# 8️⃣ Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test_onehot)
print(f"\n✅ Precisión en test: {accuracy:.4f}")

# 9️⃣ Graficar curvas de entrenamiento
plt.figure(figsize=(12, 5))

# Curva de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Pérdida entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida validación")
plt.title("Curva de Pérdida")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()

# Curva de precisión
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Precisión entrenamiento")
plt.plot(history.history["val_accuracy"], label="Precisión validación")
plt.title("Curva de Precisión")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()

plt.tight_layout()
plt.show()
