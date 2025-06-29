# Ejercicio 3 mejorado: Clasificador para predecir si x1 > x2
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# 1. Generación de datos
# =========================

# Creamos 100,000 pares de números aleatorios entre 0 y 10
X = np.random.rand(100000, 2) * 10

# Creamos la variable objetivo (y)
# y será 1 si x1 > x2, y 0 si x1 <= x2
y = (X[:, 0] > X[:, 1]).astype(int)

# =========================
# 2. Normalización
# =========================

# Las redes y modelos lineales funcionan mejor si los datos están entre 0 y 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. División en entrenamiento y prueba
# =========================

# Dividimos los datos: 80% para entrenar y 20% para probar
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 4. Entrenamiento del modelo
# =========================

# Creamos y entrenamos el modelo de clasificación logística
model = LogisticRegression()
model.fit(X_train, y_train)

# =========================
# 5. Evaluación del modelo
# =========================

# Predecimos las etiquetas de prueba
y_pred = model.predict(X_test)

# Métrica principal para clasificación: exactitud (accuracy)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy en test: {acc:.4f}")

# Mostramos la matriz de confusión
print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Mostramos un reporte completo: precisión, recall, F1
print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Mostramos los coeficientes (pesos) que aprendió el modelo
print("📈 Coeficientes:", model.coef_)
print("📈 Intercepto (bias):", model.intercept_)

# =========================
# 6. Predicciones sobre entradas conocidas
# =========================

print("\n🔍 Predicciones en casos específicos:")
ejemplos = [(2, 3), (9, 2), (5, 5)]

for d in ejemplos:
    real = int(d[0] > d[1])  # Valor real
    d_scaled = scaler.transform([d])  # Escalamos antes de predecir
    pred = model.predict(d_scaled)[0]  # Predicción
    print(f"Entrada {d} → ¿x1 > x2?: {real}, predicción del modelo: {pred}")
