# Ejercicio 4 Mejorado: Clasificador multiclase para comparar dos números

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")  # Opcional: para ocultar warnings en producción

# -----------------------------
# 1. Generación de datos base
# -----------------------------

# Generamos 100.000 pares de números aleatorios entre 0 y 10
X = np.random.rand(100000, 2) * 10
x1, x2 = X[:, 0], X[:, 1]

# Definimos una pequeña tolerancia para detectar igualdad en floats
epsilon = 1e-6

# Etiquetamos cada fila:
# 0 → x1 < x2
# 1 → x1 ≈ x2
# 2 → x1 > x2
y = np.where(np.abs(x1 - x2) < epsilon, 1, np.where(x1 > x2, 2, 0))

# -----------------------------
# 2. Balanceo de clases (casos de igualdad)
# -----------------------------

# Como x1 == x2 casi nunca ocurre con floats, agregamos manualmente 10.000 casos iguales
X_eq = np.random.rand(10000, 1) * 10
X_eq = np.hstack([X_eq, X_eq])  # segunda columna igual a la primera → x1 == x2
y_eq = np.ones(10000)  # clase 1 para estos casos

# Combinamos los datos originales con los casos forzados
X_total = np.vstack([X, X_eq])
y_total = np.concatenate([y, y_eq])

# -----------------------------
# 3. Preprocesamiento
# -----------------------------

# Escalamos los datos entre 0 y 1 para facilitar el aprendizaje del modelo
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_total)

# Dividimos en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_total, test_size=0.2, random_state=42, stratify=y_total
)

# -----------------------------
# 4. Entrenamiento del modelo
# -----------------------------

# Creamos el modelo de regresión logística (compatible con clasificación multiclase)
model = LogisticRegression(
    solver="lbfgs",  # optimizador recomendado para clasificación multiclase
    max_iter=3000,  # aumentamos iteraciones para asegurar convergencia
    random_state=42,
)

# Entrenamos con los datos
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluación del modelo
# -----------------------------

# Predicciones
y_pred = model.predict(X_test)

# Métricas generales
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy en test: {acc:.4f}")

# Matriz de confusión: muestra aciertos y errores por clase
print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Reporte completo por clase (precisión, recall, F1)
print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test, y_pred, target_names=["x1 < x2", "x1 == x2", "x1 > x2"]
    )
)

# -----------------------------
# 6. Interpretación del modelo
# -----------------------------

# Pesos aprendidos por el modelo para cada clase (coeficientes de la recta)
print("📈 Coeficientes del modelo (uno por clase):")
print(model.coef_)

# Intercepto (bias) por clase
print("📈 Intercepto (bias):")
print(model.intercept_)

# -----------------------------
# 7. Pruebas manuales
# -----------------------------

# Casos concretos para verificar la predicción del modelo
print("\n🔍 Predicciones manuales:")
ejemplos = [(2, 3), (5, 5), (9, 2)]
for d in ejemplos:
    d_scaled = scaler.transform([d])  # Escalar entrada
    pred = model.predict(d_scaled)[0]  # Predecir clase
    etiqueta = ["x1 < x2", "x1 == x2", "x1 > x2"][int(pred)]
    print(f"Entrada {d} → Clase predicha: {int(pred)} ({etiqueta})")
