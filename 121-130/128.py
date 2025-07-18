# ✅ Ejercicio 128/200 – Conversión a TFLite con optimización para CPU
import tensorflow as tf
import numpy as np

from keras.layers import Input, Dense
from keras.models import Sequential, load_model

# -------------------------
# 1. Creamos un modelo sencillo ya entrenado (simulado)
# -------------------------
# Este sería tu modelo real, aquí se usa uno mínimo para mostrar la lógica

model = Sequential(
    [
        Input(shape=(10,)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# Compila el modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entrena con datos falsos para el ejemplo
x_train = np.random.random((100, 10))
y_train = np.random.randint(0, 2, size=(100, 1))

model.fit(x_train, y_train, epochs=3, batch_size=8)

# -------------------------
# 2. Guardamos el modelo en formato .h5 (formato estándar de Keras)
# -------------------------
model.save("modelo_fake_news.h5")

# -------------------------
# 3. Convertimos a formato .tflite optimizado para CPU
# -------------------------

# Cargamos el modelo ya entrenado
model = load_model("modelo_fake_news.h5")

# Creamos el convertidor desde el modelo Keras
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Activamos la optimización para CPU con quantización post-entrenamiento
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Podemos especificar usar float16 para reducir aún más el peso
converter.target_spec.supported_types = [tf.float16]

# Convertimos el modelo
tflite_model = converter.convert()

# Guardamos el modelo tflite
with open("modelo_fake_news.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Modelo convertido y guardado como .tflite con optimización para CPU")

# -------------------------
# 4. Verificación de que el modelo tflite funciona (opcional)
# -------------------------

# Carga el modelo tflite
interpreter = tf.lite.Interpreter(model_path="modelo_fake_news.tflite")
interpreter.allocate_tensors()

# Obtén detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Crea una muestra para probar
sample_input = np.random.random((1, 10)).astype(np.float32)

# Asigna la entrada
interpreter.set_tensor(input_details[0]["index"], sample_input)

# Ejecuta la inferencia
interpreter.invoke()

# Obtiene la predicción
output_data = interpreter.get_tensor(output_details[0]["index"])

print(f"🔍 Predicción del modelo TFLite: {output_data}")
