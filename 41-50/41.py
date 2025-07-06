# 🧠 Ejercicio 41/200: Regularización con Dropout en CNN usando Fashion-MNIST
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# 1️⃣ Cargar el dataset Fashion MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2️⃣ Normalización y reestructuración de las imágenes (escalado a [0,1] y canal de 1 para CNN)
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 3️⃣ Codificación one-hot para las etiquetas (10 clases)
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# 4️⃣ Definición de arquitectura CNN con Dropout para reducir overfitting
model = Sequential(name="CNN_with_Dropout")
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))  # Apaga el 25% de neuronas aleatoriamente

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))  # Apaga otro 25% en esta capa

model.add(Flatten())  # Aplanamiento para conectar con capas densas

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))  # Regularización más fuerte en la capa densa

model.add(Dense(10, activation="softmax"))  # Clasificación de 10 clases

# 5️⃣ Compilación del modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 6️⃣ Entrenamiento del modelo con validación automática del 10%
history = model.fit(
    X_train, y_train_cat, validation_split=0.1, epochs=20, batch_size=128, verbose=1
)

# 7️⃣ Evaluación final en el conjunto de test
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"✅ Precisión en test: {acc:.4f}")

# 8️⃣ Visualización de curvas de entrenamiento
plt.figure(figsize=(12, 5))

# Curva de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Pérdida Entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida Validación")
plt.title("Curva de Pérdida con Dropout")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Curva de precisión
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Precisión Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Precisión Validación")
plt.title("Curva de Precisión con Dropout")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
