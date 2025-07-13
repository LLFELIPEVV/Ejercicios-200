# ✅ Ejercicio 78/200 — Visualización de la atención en modelos tipo Transformer
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import (
    Input,
    TextVectorization,
    Embedding,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
)
from keras.optimizers import Adam

# ----------------------------
# ⚙️ Configuración dinámica para CPUs multicore
# ----------------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

batch_size = 32 if num_threads <= 4 else 64 if num_threads <= 8 else 128

# ----------------------------
# 📥 Carga y preparación del dataset
# ----------------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# 🔠 Vectorización de texto
# ----------------------------
vectorizer = TextVectorization(
    max_tokens=5000,
    output_sequence_length=100,
    output_mode="int",
    standardize="lower_and_strip_punctuation",
)
vectorizer.adapt(X_train)

AUTOTUNE = tf.data.AUTOTUNE


# Dataset para entrenamiento y test con pipeline optimizado
def prepare_dataset(X, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(512)
    ds = ds.batch(batch_size).map(
        lambda x, y: (vectorizer(x), y), num_parallel_calls=AUTOTUNE
    )
    return ds.prefetch(AUTOTUNE)


train_ds = prepare_dataset(X_train, y_train, training=True)
test_ds = prepare_dataset(X_test, y_test, training=False)


# ----------------------------
# 🧠 Bloque Transformer con atención múltiple
# ----------------------------
def transformer_block(
    x, num_heads=2, key_dim=32, ff_dim=32, dropout_rate=0.1, block_id=0
):
    attn_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"mha_{block_id}"
    )(x, x)  # Autoatención: Query=Key=Value
    x = Add(name=f"residual_attention_{block_id}")([x, attn_output])
    x = LayerNormalization(name=f"layernorm_attn_{block_id}")(x)

    ff = Dense(ff_dim, activation="relu", name=f"ffn_{block_id}")(x)
    ff = Dropout(dropout_rate, name=f"dropout_{block_id}")(ff)
    x = Add(name=f"residual_ffn_{block_id}")([x, ff])
    x = LayerNormalization(name=f"layernorm_ffn_{block_id}")(x)

    return x


# ----------------------------
# 🧱 Construcción del modelo completo
# ----------------------------
input_layer = Input(shape=(100,), name="input_tokens")
x = Embedding(input_dim=5000, output_dim=32, name="embedding")(input_layer)

# 🔁 Apilamiento de 3 bloques Transformer
for i in range(3):
    x = transformer_block(x, block_id=i)

x = GlobalAveragePooling1D(name="pooling")(x)
x = Dropout(0.1, name="final_dropout")(x)
output = Dense(1, activation="sigmoid", name="output")(x)

model = Model(inputs=input_layer, outputs=output, name="TransformerFakeNews")
model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ----------------------------
# 🏋️ Entrenamiento del modelo
# ----------------------------
model.fit(train_ds, validation_data=test_ds, epochs=3)

# ----------------------------
# 👁️ Visualización de atención del primer bloque
# ----------------------------
# Ejemplo de texto para inspeccionar la atención
text = "The government confirmed the rumors about new policies"
tokens = vectorizer([text])
token_ids = tokens.numpy()[0]
vocab = vectorizer.get_vocabulary()
token_words = [vocab[i] for i in token_ids]

# Extraer salida de MultiHeadAttention del primer bloque
att_model = Model(inputs=model.input, outputs=model.get_layer("mha_0").output)
att_scores_out = att_model.predict(tokens)

# Visualización tipo heatmap
plt.figure(figsize=(10, 8))
plt.imshow(att_scores_out[0], cmap="viridis")
plt.xticks(ticks=range(len(token_words)), labels=token_words, rotation=90)
plt.yticks(ticks=range(len(token_words)), labels=token_words)
plt.title("Visualización de Atención — Primer bloque")
plt.colorbar()
plt.tight_layout()
plt.show()
