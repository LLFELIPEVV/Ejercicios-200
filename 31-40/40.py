# 🧠 Ejercicio 40/200: Clasificación de imágenes Fashion-MNIST usando CNN con Keras

import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1️⃣ Cargar el dataset Fashion-MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2️⃣ Preprocesamiento: escalar a [0, 1] y ajustar forma a (28, 28, 1) para CNN
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 3️⃣ Codificar etiquetas a one-hot
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# 4️⃣ Definir la arquitectura del modelo CNN
model = Sequential(
    [
        # Capa convolucional 1: 32 filtros, ventana 3x3, activación ReLU
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),  # Reducción espacial
        # Capa convolucional 2: 64 filtros
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        # Aplanar y pasar a MLP
        Flatten(),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),  # Capa oculta adicional opcional
        Dense(10, activation="softmax"),  # Capa de salida con 10 clases
    ]
)

# 5️⃣ Compilar el modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 6️⃣ Entrenar el modelo
history = model.fit(
    X_train,
    y_train_cat,
    validation_split=0.1,  # 10% para validación interna
    epochs=15,
    batch_size=128,
    verbose=1,
)

# 7️⃣ Evaluación final sobre el set de prueba
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"✅ Precisión en test: {acc:.4f}")

# 8️⃣ Visualización de curvas de entrenamiento
plt.figure(figsize=(12, 5))

# Curva de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Pérdida entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida validación")
plt.title("Curva de pérdida")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()

# Curva de precisión
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Precisión entrenamiento")
plt.plot(history.history["val_accuracy"], label="Precisión validación")
plt.title("Curva de precisión")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()

plt.tight_layout()
plt.show()
