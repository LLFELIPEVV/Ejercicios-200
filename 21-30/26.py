# 📚 Librerías necesarias
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Cargar y preparar los datos
print("📥 Cargando datos del conjunto Iris...")
iris = load_iris()
X, y = iris.data, iris.target
nombres_clases = iris.target_names

# 2️⃣ Escalar los datos (muy importante para redes neuronales)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Definir la grilla de hiperparámetros para explorar
param_grid = {
    "hidden_layer_sizes": [
        (2,),
        (4,),
        (4, 2),
        (8,),
        (8, 4),
        (8, 4, 2),
        (16,),
        (16, 8),
        (16, 8, 4),
        (16, 8, 4, 2),
        (32,),
        (32, 16),
        (32, 16, 8),
        (32, 16, 8, 4),
        (32, 16, 8, 4, 2),
        (64,),
        (64, 32),
        (64, 32, 16),
        (64, 32, 16, 8),
        (64, 32, 16, 8, 4),
        (64, 32, 16, 8, 4, 2),
    ],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2],  # Regularización L2
    "activation": ["relu", "tanh"],  # Función de activación
}

# 5️⃣ Crear el modelo base
mlp = MLPClassifier(
    solver="adam",  # Optimizador adaptativo eficiente
    max_iter=3000,  # Suficientes iteraciones para converger
    random_state=42,
)

# 6️⃣ Buscar la mejor combinación de hiperparámetros usando validación cruzada
print("\n🔍 Ejecutando búsqueda de hiperparámetros...")
grid = GridSearchCV(
    estimator=mlp, param_grid=param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=2
)
grid.fit(X_train, y_train)

# 7️⃣ Resultados de la búsqueda
print(f"\n🏆 Mejor accuracy validación cruzada: {grid.best_score_:.4f}")
print("🔧 Mejores hiperparámetros encontrados:")
for param, value in grid.best_params_.items():
    print(f"   {param}: {value}")

# 8️⃣ Evaluar el modelo óptimo sobre el conjunto de prueba
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# 9️⃣ Métricas de evaluación
acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred, target_names=nombres_clases)

print(f"\n✅ Accuracy final sobre test: {acc:.4f}")
print("\n📋 Reporte de clasificación:")
print(rep)

# 🔟 Visualización: Matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(
    conf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=nombres_clases,
    yticklabels=nombres_clases,
)
plt.title("Matriz de confusión - Clasificación Iris con MLP")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Real")
plt.tight_layout()
plt.show()
