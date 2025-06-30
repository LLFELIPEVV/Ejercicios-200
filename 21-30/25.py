# 🧪 Ejercicio 25: Optimización de red neuronal MLP con GridSearchCV sobre MNIST
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Cargar el dataset MNIST desde OpenML
print("📥 Cargando datos de MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 2️⃣ Escalado de características: normalizar a rango [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ División de datos con estratificación para mantener proporciones por clase
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Definición de la grilla de búsqueda de hiperparámetros
param_grid = {
    "hidden_layer_sizes": [(64,), (64, 64), (128,), (256, 128), (256, 128, 64)],
    "alpha": [0.0001, 0.001, 0.01],  # Regularización L2
    "activation": ["relu", "tanh"],  # Función de activación
}

# 5️⃣ Definición del modelo base (no entrenado aún)
mlp = MLPClassifier(
    solver="adam",  # Optimizador eficiente
    batch_size=256,  # Tamaño de minibatch
    max_iter=100,  # Iteraciones máximas para entrenar
    random_state=42,  # Reproducibilidad
)

# 6️⃣ Aplicar Grid Search con validación cruzada
print("🔍 Iniciando búsqueda de hiperparámetros con GridSearchCV...")
grid = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,  # Validación cruzada 3-fold
    n_jobs=-1,  # Paralelizar en todos los núcleos
    verbose=2,  # Mostrar progreso detallado
)

grid.fit(X_train, y_train)

# 7️⃣ Mostrar mejores resultados encontrados
print(f"\n🏆 Mejor accuracy validación cruzada: {grid.best_score_:.4f}")
print("🔧 Mejores hiperparámetros encontrados:")
for param, value in grid.best_params_.items():
    print(f"   {param}: {value}")

# 8️⃣ Evaluar el mejor modelo sobre conjunto de prueba
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred)

print(f"\n✅ Accuracy sobre el test set: {acc:.4f}")
print("\n📋 Reporte de clasificación (por clase):")
print(rep)

print("\n📊 Matriz de confusión:")
print(conf)

# 9️⃣ Visualizar algunas predicciones correctamente clasificadas
print("\n🖼 Visualizando ejemplos del conjunto de prueba:")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = X_test[i].reshape(28, 28)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"P:{y_pred[i]} / R:{y_test[i]}")
    ax.axis("off")

plt.suptitle("Predicciones del mejor modelo (GridSearch)")
plt.tight_layout()
plt.show()
