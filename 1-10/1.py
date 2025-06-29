# Ejercicio 1: Modelo de Machine Learning para aprender a sumar dos números

# Importamos las librerías necesarias
import numpy as np  # Para generar datos aleatorios y trabajar con arrays
from sklearn.metrics import mean_squared_error  # Para evaluar el modelo
from sklearn.linear_model import LinearRegression  # Modelo de regresión lineal
from sklearn.model_selection import (
    train_test_split,
)  # Para dividir los datos en entrenamiento y prueba

# ============================
# 1. Generación de datos
# ============================

# Creamos una matriz X de 1000 filas y 2 columnas con números aleatorios entre 0 y 10
X = np.random.rand(1000, 2) * 10

# Calculamos la suma de cada par de números (por fila), que será nuestra variable objetivo (y)
y = X.sum(axis=1)

# ============================
# 2. División del dataset
# ============================

# Dividimos los datos: 80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# 3. Entrenamiento del modelo
# ============================

# Creamos el modelo de regresión lineal
model = LinearRegression()

# Entrenamos el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# ============================
# 4. Evaluación del modelo
# ============================

# Hacemos predicciones con los datos de prueba
y_pred = model.predict(X_test)

# Calculamos el error cuadrático medio (qué tan lejos están las predicciones del valor real)
mse = mean_squared_error(y_test, y_pred)

# Mostramos el error, los coeficientes (pesos) aprendidos y el sesgo (bias/intercepto)
print(f"MSE en test: {mse:.4f}")
print("Coeficientes:", model.coef_, "Bias:", model.intercept_)

# ============================
# 5. Predicciones de prueba
# ============================

# Probamos el modelo con ejemplos concretos para ver si aprendió a sumar
for d in [(2, 3), (7.5, 1.2), (0, 0)]:
    print(f"Entrada {d}, suma real {sum(d)}, predicción {model.predict([d])[0]:.2f}")
