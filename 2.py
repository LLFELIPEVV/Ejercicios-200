import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# ========================
# 1. Generar datos sintéticos
# ========================

# Creamos 100,000 pares de números aleatorios entre 0 y 10
X = np.random.rand(100000, 2) * 10

# Calculamos la multiplicación (producto) de cada par, eje=1 significa fila a fila
y = np.prod(X, axis=1)

# ========================
# 2. Normalización
# ========================

# Las redes neuronales aprenden mejor si los datos están entre 0 y 1
# Usamos MinMaxScaler para escalar los datos de entrada
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 3. Dividir datos en entrenamiento y prueba
# ========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ========================
# 4. Crear el modelo
# ========================

model = MLPRegressor(
    hidden_layer_sizes=(50, 50),  # Dos capas ocultas de 50 neuronas cada una
    activation="relu",  # ReLU: permite que el modelo aprenda funciones no lineales
    solver="adam",  # Algoritmo de optimización moderno y eficiente
    max_iter=20000,  # Aumentamos el número de iteraciones para asegurar el aprendizaje
    random_state=42,  # Semilla para reproducibilidad
)

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# ========================
# 5. Evaluar el modelo
# ========================

# Hacemos predicciones sobre los datos de prueba
y_pred = model.predict(X_test)

# Calculamos el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE en test: {mse:.4f}")

# ========================
# 6. Pruebas con datos reales
# ========================

# Pruebas con números conocidos
# IMPORTANTE: ¡Debemos escalar también los datos de entrada para predecir!
test_values = [(2, 3), (7.5, 1.2), (0, 10)]

for d in test_values:
    d_scaled = scaler.transform([d])  # Escalar los valores antes de predecir
    real = d[0] * d[1]  # Resultado real
    pred = model.predict(d_scaled)[0]  # Predicción del modelo
    print(
        f"Entrada {d}, multiplicación real {real:.2f}, predicción del modelo {pred:.2f}"
    )
