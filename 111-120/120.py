# 🧪 Ejercicio 120/200 — Exportación del modelo a .h5 y .tflite optimizado para CPU
import re
import time
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense

data = {
    "text": [
        "New vaccine approved by government",
        "Aliens take over White House!",
        "Experts warn about climate change",
        "Cure for cancer found in bananas!",
        "Economy improves, says president",
        "Drinking cola extends your life",
    ],
    "label": [0, 1, 0, 1, 0, 1],  # 0: real, 1: fake
}
df = pd.DataFrame(data)


# 3. 🧼 Función para limpiar los textos
def clean_text(text):
    text = text.lower()  # Poner todo en minúsculas
    text = re.sub(r"http\S+|www\S+", "", text)  # Eliminar URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Quitar números y signos
    return text.strip()  # Quitar espacios al inicio y al final


# Aplicar limpieza a cada texto
df["text"] = df["text"].apply(clean_text)

# 4. 🧪 Separar los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.33, random_state=42
)

# 5. 🧠 Vectorización del texto (convertir palabras a números)
vectorizer = TextVectorization(
    max_tokens=1000,  # Número máximo de palabras
    output_mode="int",  # Salida como números enteros
    output_sequence_length=20,  # Todas las frases tendrán 20 palabras (rellenadas o recortadas)
)
vectorizer.adapt(X_train)  # Aprende el vocabulario a partir de los textos

# 6. ⚙️ Preparar los datos con tf.data.Dataset (optimizado para CPU)
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .batch(2)  # Entrenamiento en grupos de 2
    .cache()  # Guarda en memoria RAM
    .prefetch(tf.data.AUTOTUNE)  # Carga anticipada para mejorar velocidad
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(2)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# 7. 🧠 Crear un modelo simple y eficiente
model = Sequential(
    [
        vectorizer,  # Paso 1: convierte texto en números
        Embedding(
            input_dim=1000, output_dim=16
        ),  # Paso 2: crea vectores para cada palabra
        GlobalAveragePooling1D(),  # Paso 3: promedia todos los vectores de una oración
        Dense(1, activation="sigmoid"),  # Paso 4: predice si es real (0) o falsa (1)
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 8. 🛑 Añadir callbacks para optimizar el entrenamiento
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1)

# 9. 🏋️ Entrenar el modelo
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# 10. 📊 Visualización del entrenamiento con seaborn
history_df = pd.DataFrame(history.history)

# Guarda el modelo en un archivo .h5 (formato clásico de Keras)
model.save("modelo_fake_news.h5")
print("✅ Modelo guardado como modelo_fake_news.h5")

# Crea un convertidor de TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Habilita optimizaciones para CPU de bajo consumo
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convierte el modelo
tflite_model = converter.convert()

# Guarda el archivo convertido
with open("modelo_fake_news.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Modelo convertido y guardado como modelo_fake_news.tflite")

# Cargar modelo TFLite en un intérprete
interpreter = tf.lite.Interpreter(model_path="modelo_fake_news.tflite")
interpreter.allocate_tensors()

# Obtener información de entrada/salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Texto de ejemplo a predecir
texto = ["aliens landed at the white house"]
tensor_input = tf.convert_to_tensor(texto)

# Usa el mismo vectorizador que fue entrenado
vector_input = model.layers[0](tensor_input)

# Simular inferencia
interpreter.set_tensor(input_details[0]["index"], vector_input.numpy())
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])

print("🔮 Predicción TFLite:", output_data[0][0])

# Comparar con modelo original
start = time.time()
_ = model.predict(texto)
end = time.time()
print(f"⏱ Tiempo con modelo Keras: {end - start:.5f} segundos")

# Tiempo con modelo TFLite
start = time.time()
interpreter.set_tensor(input_details[0]["index"], vector_input.numpy())
interpreter.invoke()
_ = interpreter.get_tensor(output_details[0]["index"])
end = time.time()
print(f"⏱ Tiempo con modelo TFLite: {end - start:.5f} segundos")
