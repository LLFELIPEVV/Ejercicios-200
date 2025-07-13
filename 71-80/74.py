# ✅ Ejercicio 74/200 — Introducción al Mecanismo de Atención en Redes Neuronales para Fake News
# ======================================
# 🧠 Clasificador de Fake News con Atención
# ======================================
import pandas as pd
import tensorflow as tf

from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    TextVectorization,
    Layer,
)
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------------------------
# 📥 1. Carga y Preprocesamiento de Datos
# ----------------------------

# Cargar datasets etiquetados
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0  # Etiqueta para noticias falsas
df_true["label"] = 1  # Etiqueta para noticias reales

# Concatenar y limpiar
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()

# Separar características y etiquetas
X, y = df["text"].values, df["label"].values

# Dividir en conjunto de entrenamiento y prueba (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ----------------------------
# 🧹 2. Vectorización de texto
# ----------------------------

# Convierte texto en secuencias numéricas con padding a 300 tokens
vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=300)
vectorizer.adapt(X_train)  # Aprende el vocabulario del entrenamiento

# Transformar los textos
X_train_seq = vectorizer(X_train)
X_test_seq = vectorizer(X_test)

# ----------------------------
# 🎯 3. Capa de Atención Personalizada
# ----------------------------


class Attention(Layer):
    """
    Capa de atención simple que aprende pesos de importancia por palabra.
    """

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),  # Dimensión del vector oculto
            initializer="normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),  # Tamaño de secuencia
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        # Calcular importancia de cada palabra
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)  # Pesos de atención
        output = inputs * a  # Multiplicar entrada por atención
        return tf.keras.backend.sum(output, axis=1)  # Sumar con atención aplicada


# ----------------------------
# 🧱 4. Arquitectura del Modelo
# ----------------------------

input_layer = Input(shape=(300,), name="input_sequence")
x = Embedding(input_dim=10000, output_dim=128, name="embedding")(input_layer)
x = Bidirectional(LSTM(64, return_sequences=True), name="bi_lstm")(x)
x = Attention()(x)
output = Dense(1, activation="sigmoid", name="output")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ----------------------------
# 🏋️ 5. Entrenamiento
# ----------------------------

model.fit(
    X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
)

# ----------------------------
# 📊 6. Evaluación del Modelo
# ----------------------------

y_pred = model.predict(X_test_seq).flatten()
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n📈 Reporte de clasificación:\n")
print(classification_report(y_test, y_pred_labels, zero_division=0))
