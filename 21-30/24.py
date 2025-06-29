# 🧪 Ejercicio 24: Clasificación de dígitos MNIST con red neuronal MLP (modelo profundo)
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Carga del dataset MNIST desde OpenML
print("📥 Cargando datos de MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)  # Etiquetas convertidas a enteros

# 2️⃣ Preprocesamiento: Normalización de píxeles entre 0 y 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Cada pixel entre [0, 1]

# 3️⃣ División de los datos en entrenamiento y prueba (estratificado para balancear clases)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Definición del modelo MLP (Multilayer Perceptron)
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # Tres capas ocultas con muchas neuronas
    activation="relu",  # Función de activación no lineal moderna
    solver="adam",  # Optimizador eficiente para datasets grandes
    batch_size=256,  # Procesa 256 ejemplos por lote (reduce memoria y acelera)
    max_iter=50,  # Número de épocas
    random_state=42,  # Reproducibilidad
    verbose=True,  # Muestra progreso durante el entrenamiento
)

# 5️⃣ Entrenamiento del modelo
print("\n🚀 Entrenando red neuronal MLP...")
model.fit(X_train, y_train)

# 6️⃣ Evaluación del modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred)

# 7️⃣ Resultados
print(f"\n✅ Accuracy en test: {acc:.4f}")
print("\n📋 Reporte de clasificación por clase:")
print(rep)
print("\n📊 Matriz de confusión:")
print(conf)

# 8️⃣ Visualización de algunas predicciones
print("\n🖼 Visualizando algunas predicciones:")
plt.figure(figsize=(12, 5))
for i in range(10):
    imagen = X_test[i].reshape(28, 28)
    plt.subplot(2, 5, i + 1)
    plt.imshow(imagen, cmap="gray")
    plt.title(f"Pred: {y_pred[i]}\nReal: {y_test[i]}")
    plt.axis("off")

plt.suptitle("Predicciones del modelo - Dígitos MNIST")
plt.tight_layout()
plt.show()
