# 🧪 Ejercicio 116/200 — Exportación del modelo a .h5 y .tflite optimizado para CPU
# 1. Librerías necesarias
import os
import re
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense

# 2. Datos de ejemplo (Fake = 1, Real = 0)
data = {
    "text": [
        "Government launches new education policy",
        "Aliens spotted near the Eiffel Tower!",
        "Health department issues new pandemic warning",
        "Cure for aging discovered!",
    ],
    "label": [0, 1, 0, 1],
}
df = pd.DataFrame(data)


# 3. Limpieza de texto con expresiones regulares
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # eliminar URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # eliminar signos y números
    return text.strip()


df["text"] = df["text"].apply(clean_text)

# 4. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.5, random_state=42
)

# 5. Vectorizador
vectorizer = TextVectorization(
    max_tokens=1000, output_mode="int", output_sequence_length=20
)
vectorizer.adapt(X_train)


# 6. Preparación con tf.data.Dataset optimizado
def vectorize(text, label):
    return vectorizer(text), label


train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(vectorize)
    .batch(2)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .map(vectorize)
    .batch(2)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# 7. Modelo ligero
model = Sequential(
    [
        Embedding(input_dim=1000, output_dim=16),
        GlobalAveragePooling1D(),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 8. Entrenamiento con EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

model.fit(
    train_ds, validation_data=test_ds, epochs=20, callbacks=[early_stop], verbose=1
)

# 9. Guardar modelo como .h5
model.save("modelo_fake_news.h5")
print("Modelo guardado como modelo_fake_news.h5")

# 10. Convertir a .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar en archivo
with open("modelo_fake_news.tflite", "wb") as f:
    f.write(tflite_model)
print("Modelo convertido y guardado como modelo_fake_news.tflite")

# 11. Verificar tamaño del modelo TFLite (opcional)
size = os.path.getsize("modelo_fake_news.tflite") / 1024
print(f"Tamaño del modelo .tflite: {size:.2f} KB")

# 12. Cargar y evaluar modelo TFLite (opcional para validación)
# Requiere preparar un intérprete

interpreter = tf.lite.Interpreter(model_path="modelo_fake_news.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Usamos un ejemplo procesado
example = tf.constant(["aliens spotted!"])  # texto de prueba
example_vec = vectorizer(example)
example_vec = tf.cast(example_vec, dtype=tf.float32)  # ✅ Convertir a float32

interpreter.set_tensor(input_details[0]["index"], example_vec.numpy())
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

print(
    "Predicción TFLite:", prediction, "| Clase:", "Fake" if prediction > 0.5 else "Real"
)
