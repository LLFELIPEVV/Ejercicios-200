# Ejercicio 5: Clasificador binario para saber si un punto está sobre o por encima de la recta y = 2x + 3

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings(
    "ignore"
)  # Opcional: ocultar advertencias si estás en modo presentación o pruebas

# ---------------------------------------------
# 1. Generación del dataset sintético
# ---------------------------------------------

# Creamos 100.000 puntos aleatorios en el plano (x, y), entre 0 y 10
X = np.random.rand(100000, 2) * 10

# Separamos las coordenadas x e y para trabajar más cómodamente
x_vals = X[:, 0]
y_vals = X[:, 1]

# Regla de clasificación:
# Clase 1 → si el punto está por encima de la línea y = 2x + 3
# Clase 0 → si está por debajo o sobre la línea
y = (y_vals > 2 * x_vals + 3).astype(int)

# ---------------------------------------------
# 2. Escalado de características (normalización)
# ---------------------------------------------

# Escalamos los valores entre 0 y 1 para mejorar el rendimiento del modelo
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------
# 3. División del dataset
# ---------------------------------------------

# Dividimos en 80% entrenamiento y 20% test. Stratify asegura proporción balanceada de clases.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------
# 4. Creación y entrenamiento del modelo
# ---------------------------------------------

# Usamos regresión logística (modelo lineal) para clasificación binaria
model = LogisticRegression(
    solver="lbfgs",  # optimizador eficiente recomendado por sklearn
    max_iter=2000,  # iteraciones suficientes para asegurar convergencia
    random_state=42,
)
model.fit(X_train, y_train)

# ---------------------------------------------
# 5. Evaluación del modelo
# ---------------------------------------------

# Realizamos predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Métrica de precisión global (accuracy)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy del modelo: {acc:.4f}")

# Reporte detallado: precisión, recall, F1-score por clase
print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred, target_names=["Abajo/Sobre la línea", "Encima de la línea"]
    )
)

# Matriz de confusión: muestra cuántos aciertos y errores hubo por clase
print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------
# 6. Pruebas manuales con nuevos puntos
# ---------------------------------------------

print("\n🔍 Predicciones manuales:")

# Lista de puntos a probar manualmente
ejemplos = [(2, 9), (1, 4), (3, 3), (5, 14)]

# Iteramos y clasificamos cada punto
for punto in ejemplos:
    punto_scaled = scaler.transform(
        [punto]
    )  # Escalamos igual que los datos de entrenamiento
    pred = model.predict(punto_scaled)[0]  # Obtenemos la predicción (0 o 1)
    etiqueta = "Encima de la línea" if pred == 1 else "Abajo o sobre la línea"
    print(f"Punto {punto} → Clase predicha: {pred} ({etiqueta})")
