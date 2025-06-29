# 🧪Ejercicio 23: Clasificación de imágenes de ropa (Fashion-MNIST) con MLP
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔸 1. Cargar el dataset Fashion MNIST (70,000 imágenes de ropa en 28x28 pixeles)
print("📥 Cargando datos de Fashion MNIST...")
mnist_fashion = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
X, y = mnist_fashion.data, mnist_fashion.target

# 🔸 2. Normalizar los pixeles a rango [0, 1] para acelerar el aprendizaje
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🔸 3. Dividir el conjunto de datos en entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # Mantiene proporciones de clases iguales en train y test
)

# 🔸 4. Definir y entrenar la red neuronal MLP
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Dos capas ocultas: 128 y 64 neuronas
    activation="relu",  # Función de activación ReLU
    solver="adam",  # Optimizador eficiente para grandes datasets
    max_iter=50,  # Iteraciones (puedes aumentarlas si no converge)
    random_state=42,
    verbose=True,  # Muestra el progreso del entrenamiento
)

print("🚀 Entrenando modelo...")
model.fit(X_train, y_train)

# 🔸 5. Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
rep = classification_report(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

print(f"\n✅ Accuracy del modelo: {acc:.4f}")
print("\n📋 Reporte de clasificación por clase:")
print(rep)
print("\n📊 Matriz de confusión:")
print(conf)

# 🔸 6. Visualizar algunas imágenes y sus predicciones
clase_nombres = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

print("\n🖼 Visualizando predicciones:")
plt.figure(figsize=(12, 5))

for i in range(10):
    imagen = X_test[i].reshape(28, 28)  # Las imágenes están aplanadas (784,)
    pred = y_pred[i]
    real = y_test[i]

    plt.subplot(2, 5, i + 1)
    plt.imshow(imagen, cmap="gray")
    plt.title(f"Pred: {clase_nombres[int(pred)]}\nReal: {clase_nombres[int(real)]}")
    plt.axis("off")

plt.suptitle("Predicciones del modelo - Imágenes de ropa")
plt.tight_layout()
plt.show()
