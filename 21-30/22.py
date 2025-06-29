# 🧪 Ejercicio 22: Clasificación de letras del abecedario usando red neuronal MLP
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Cargar dataset de letras manuscritas (26 clases: A-Z)
print("📥 Cargando dataset 'letter' de OpenML...")
letras = fetch_openml("letter", version=1, as_frame=False)
X, y = letras.data, letras.target  # X: (20000, 16), y: letras (strings de 'A' a 'Z')

# 2️⃣ Normalizar los datos a rango [0, 1] para mejorar el entrenamiento
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Separar en conjuntos de entrenamiento y prueba (80/20), usando estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # 📌 Asegura proporciones balanceadas por clase
)

# 4️⃣ Definir y entrenar modelo MLP (Red Neuronal Multicapa)
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Dos capas ocultas: 128 y 64 neuronas
    activation="relu",  # Función de activación no lineal ReLU
    solver="adam",  # Optimizador adaptativo Adam
    max_iter=3000,  # Máximo de iteraciones de entrenamiento
    random_state=42,
    verbose=True,  # Muestra progreso de entrenamiento
)

print("🚀 Entrenando modelo MLP...")
model.fit(X_train, y_train)

# 5️⃣ Evaluación del modelo en el conjunto de prueba
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy en test: {acc:.4f}")
print("\n📋 Reporte de clasificación (por letra):")
print(classification_report(y_test, y_pred, digits=4))

print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 6️⃣ Visualizar predicciones (representación simplificada tipo imagen 4x4)
print("\n🖼 Ejemplos de letras y predicciones:")
plt.figure(figsize=(10, 4))
for i in range(10):
    # Cada muestra tiene 16 features → lo representamos como una matriz 4x4 solo como referencia
    plt.subplot(2, 5, i + 1)
    letra = y_pred[i]
    muestra = X_test[i].reshape(4, 4)
    plt.imshow(muestra, cmap="binary")
    plt.title(f"Pred: {letra}")
    plt.axis("off")

plt.suptitle("Visualización simple (4x4) de letras clasificadas")
plt.tight_layout()
plt.show()
