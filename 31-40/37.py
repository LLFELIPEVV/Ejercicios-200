# 🧠 Ejercicio 37/200: Visualización del entrenamiento de un MLP (Keras) en Iris
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# 1️⃣ Cargar dataset y variables
iris = load_iris()
X, y = iris.data, iris.target
n_clases = len(set(y))  # = 3

# 2️⃣ Escalado y codificación
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_encoded = to_categorical(y, num_classes=n_clases)

# 3️⃣ División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y, random_state=42
)

# 4️⃣ Definición del modelo MLP
model = Sequential(
    [
        Dense(16, activation="relu", input_shape=(X.shape[1],)),  # capa de entrada
        Dense(8, activation="relu"),  # capa oculta intermedia
        Dense(4, activation="relu"),  # capa oculta reducida
        Dense(2, activation="relu"),  # cuello de botella
        Dense(n_clases, activation="softmax"),  # salida softmax (3 clases)
    ]
)

# 5️⃣ Compilación: define cómo entrenar
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 6️⃣ Entrenamiento: history guarda todo el historial de métricas
history = model.fit(
    X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1
)

# 7️⃣ Visualización de métricas
plt.figure(figsize=(12, 5))

# 🔴 Curva de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.xlabel("Epochs")
plt.ylabel("Pérdida (Loss)")
plt.title("Evolución de la pérdida")
plt.legend()

# 🔵 Curva de precisión
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.xlabel("Epochs")
plt.ylabel("Precisión")
plt.title("Evolución de la precisión")
plt.legend()

plt.tight_layout()
plt.show()
