# Ejercicio 6: Clasificador binario que detecta si un punto está dentro de un círculo

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Ignorar advertencias de sklearn si se muestran durante entrenamiento
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1. Generación del dataset sintético
# -------------------------------------------------

# Generamos 100.000 puntos (x, y) aleatorios entre 0 y 10
X = np.random.rand(100000, 2) * 10

# Definimos el círculo: centro (5, 5), radio 3
cx, cy = 5, 5
r = 3

# Calculamos la distancia cuadrada al centro: (x - cx)^2 + (y - cy)^2
distancia_cuadrada = (X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2

# Clase 1: dentro del círculo, Clase 0: fuera del círculo
y = (distancia_cuadrada < r**2).astype(int)

# -------------------------------------------------
# 2. Preprocesamiento: Normalización de datos
# -------------------------------------------------

# Escalamos las características entre 0 y 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# 3. División del dataset
# -------------------------------------------------

# Separamos entrenamiento y prueba (80/20), asegurando proporciones con stratify
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# 4. Creación y entrenamiento del modelo
# -------------------------------------------------

# Usamos regresión logística, modelo lineal para clasificación binaria
model = LogisticRegression(solver="lbfgs", max_iter=3000, random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------
# 5. Evaluación del modelo
# -------------------------------------------------

# Predicción en los datos de prueba
y_pred = model.predict(X_test)

# Precisión global del modelo
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy del modelo: {acc:.4f}")

# Reporte detallado: precisión, recall, f1-score
print("\n📋 Reporte de clasificación:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fuera del círculo", "Dentro del círculo"],
        zero_division=0,
    )
)

# Matriz de confusión para verificar aciertos/errores por clase
print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------
# 6. Pruebas manuales para evaluar puntos específicos
# -------------------------------------------------

print("\n🔍 Predicciones para puntos manuales:")

# Lista de puntos que probaremos manualmente
ejemplos = [(5, 5), (2, 2), (8, 5), (5, 2), (6, 8)]

for punto in ejemplos:
    # Escalamos el punto usando el mismo scaler del entrenamiento
    punto_scaled = scaler.transform([punto])
    # Predecimos su clase
    pred = model.predict(punto_scaled)[0]
    etiqueta = "Dentro del círculo" if pred == 1 else "Fuera del círculo"
    print(f"Punto {punto} → Clase predicha: {pred} ({etiqueta})")
