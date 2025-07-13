# ✅ Ejercicio 79/200 — Comparativa: LSTM vs Self-Attention para clasificación de Fake News
import os
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Model, Sequential
from keras.layers import (
    Input,
    Embedding,
    LSTM,
    Bidirectional,
    Dropout,
    Dense,
    TextVectorization,
    MultiHeadAttention,
    Add,
    LayerNormalization,
    GlobalAveragePooling1D,
)

# --------------------------------------------------
# ⚙️ Configuración dinámica del entorno para CPU
# --------------------------------------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

batch_size = 32 if num_threads <= 4 else 64 if num_threads <= 8 else 128

# --------------------------------------------------
# 📥 1. Carga y preparación de datos
# --------------------------------------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# --------------------------------------------------
# 🧹 2. Tokenización y dataset eficiente
# --------------------------------------------------
vectorizer = TextVectorization(
    max_tokens=5000,
    output_sequence_length=100,
    standardize="lower_and_strip_punctuation",
    output_mode="int",
)
vectorizer.adapt(X_train)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(512)
    .batch(batch_size)
    .map(lambda x, y: (vectorizer(x), y), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(batch_size)
    .map(lambda x, y: (vectorizer(x), y), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# --------------------------------------------------
# 🔁 3. Modelo clásico: Bidirectional LSTM
# --------------------------------------------------
lstm_model = Sequential(
    [
        Input(shape=(100,), name="input_lstm"),
        Embedding(input_dim=5000, output_dim=32, name="embedding_lstm"),
        Bidirectional(LSTM(32), name="bilstm"),
        Dropout(0.1, name="dropout_lstm"),
        Dense(1, activation="sigmoid", name="output_lstm"),
    ]
)
lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# --------------------------------------------------
# 🧠 4. Modelo moderno: Self-Attention tipo Transformer
# --------------------------------------------------
def transformer_block(x, num_heads=2, key_dim=32, ff_dim=32, dropout_rate=0.1):
    attention_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = Add()([x, attention_out])  # Skip connection
    x = LayerNormalization()(x)

    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dropout(dropout_rate)(ffn)
    x = Add()([x, ffn])  # Otro skip connection
    return LayerNormalization()(x)


# Construcción del modelo basado en atención
input_att = Input(shape=(100,), name="input_transformer")
x = Embedding(input_dim=5000, output_dim=32, name="embedding_transformer")(input_att)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
output_att = Dense(1, activation="sigmoid", name="output_transformer")(x)

attention_model = Model(input_att, output_att)
attention_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# --------------------------------------------------
# 🏋️ 5. Entrenamiento de ambos modelos
# --------------------------------------------------
print("\n🚀 Entrenando modelo LSTM...")
lstm_model.fit(train_ds, validation_data=test_ds, epochs=3)

print("\n🚀 Entrenando modelo Self-Attention...")
attention_model.fit(train_ds, validation_data=test_ds, epochs=3)

# --------------------------------------------------
# 📊 6. Evaluación comparativa
# --------------------------------------------------
print("\n📈 Evaluación del modelo LSTM:")
y_pred_lstm = lstm_model.predict(test_ds).flatten()
print(classification_report(y_test, (y_pred_lstm > 0.5).astype(int), zero_division=0))

print("\n📈 Evaluación del modelo Self-Attention:")
y_pred_att = attention_model.predict(test_ds).flatten()
print(classification_report(y_test, (y_pred_att > 0.5).astype(int), zero_division=0))
