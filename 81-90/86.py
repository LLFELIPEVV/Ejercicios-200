# 🧪 Ejercicio 86/200 — Visualización de pesos y atención en un modelo Transformer para clasificación de fake news (Keras)
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    Dense,
    Dropout,
    Add,
    GlobalAveragePooling1D,
    LayerNormalization,
    MultiHeadAttention,
)
from keras.optimizers import Adam
from keras.layers import TextVectorization
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# ⚙️ Configuración eficiente
# --------------------------------------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

# --------------------------------------------------
# 📥 Carga de datos
# --------------------------------------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true])[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# --------------------------------------------------
# 🧹 Vectorización del texto
# --------------------------------------------------
vectorizer = TextVectorization(
    max_tokens=5000,
    output_sequence_length=50,
    standardize="lower_and_strip_punctuation",
)
vectorizer.adapt(X_train)
vocab = vectorizer.get_vocabulary()
vocab_dict = {i: token for i, token in enumerate(vocab)}

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(256)
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
# 🔍 Capa personalizada para atención con acceso a scores
# --------------------------------------------------
class AttentionWithScores(tf.keras.layers.Layer):
    def __init__(self, num_heads=2, key_dim=32, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )
        self.last_attention_scores = None

    def call(self, query, value, training=False):
        output, scores = self.attn(
            query, value, return_attention_scores=True, training=training
        )
        self.last_attention_scores = scores  # se almacena para uso posterior
        return output

    @property
    def scores(self):
        """Propiedad para acceder a los scores de atención"""
        return self.last_attention_scores


# --------------------------------------------------
# 🧱 Modelo con capa de atención visualizable
# --------------------------------------------------
def build_model():
    input_layer = Input(shape=(50,), dtype=tf.int32)
    x = Embedding(input_dim=5000, output_dim=32)(input_layer)

    # Capa de atención personalizada
    attn_layer = AttentionWithScores(num_heads=2, key_dim=32)
    attn_output = attn_layer(x, x)

    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    ff = Dense(32, activation="relu")(x)
    ff = Dropout(0.1)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output)
    return model, attn_layer


model, attn_layer = build_model()
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# --------------------------------------------------
# 🏋️ Entrenamiento
# --------------------------------------------------
model.fit(train_ds, validation_data=test_ds, epochs=3, verbose=1)

# --------------------------------------------------
# 🔬 Visualización de atención en ejemplo
# --------------------------------------------------
sample_text = X_test[42]
print("\n📰 Texto de prueba:", sample_text)

# Vectorizamos manualmente
tokenized = vectorizer(tf.constant([sample_text]))

# Ejecutamos la predicción para activar la capa de atención
prediction = model(tokenized, training=False)
print(f"Predicción: {prediction.numpy()[0][0]:.4f}")

# Ahora accedemos a los scores de atención
attention_scores = attn_layer.scores
if attention_scores is not None:
    attn_scores = attention_scores.numpy()[0]  # [heads, seq_len, seq_len]
    attention_map = attn_scores.mean(axis=0)[
        0
    ]  # Promedio entre heads, atención del primer token
else:
    print("⚠️ No se pudieron obtener los scores de atención")
    attention_map = None

# Mapeamos tokens
tokens = [vocab_dict.get(idx, "") for idx in tokenized.numpy()[0]]

# Filtramos tokens vacíos y ajustamos la longitud
valid_tokens = [(i, token) for i, token in enumerate(tokens) if token.strip()]
valid_indices = [i for i, token in valid_tokens]
valid_token_names = [token for i, token in valid_tokens]

# Solo creamos las visualizaciones si tenemos scores de atención
if attention_map is not None:
    # Ajustamos el mapa de atención a los tokens válidos
    attention_values = attention_map[: len(valid_indices)]

    # Gráfica
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(valid_token_names)), attention_values)
    plt.xticks(
        range(len(valid_token_names)), valid_token_names, rotation=45, ha="right"
    )
    plt.title("Atención promedio sobre los tokens")
    plt.ylabel("Peso de atención")
    plt.xlabel("Tokens")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 🔬 Visualización adicional: mapa de calor de atención
    # --------------------------------------------------
    plt.figure(figsize=(10, 8))
    # Tomamos solo los primeros N tokens para visualización
    max_tokens = min(20, len(valid_token_names))
    attn_matrix = attn_scores.mean(axis=0)[:max_tokens, :max_tokens]

    plt.imshow(attn_matrix, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Peso de atención")
    plt.title("Mapa de calor de atención entre tokens")
    plt.xlabel("Tokens (destino)")
    plt.ylabel("Tokens (origen)")

    # Etiquetas en los ejes
    token_labels = valid_token_names[:max_tokens]
    plt.xticks(range(max_tokens), token_labels, rotation=45, ha="right")
    plt.yticks(range(max_tokens), token_labels)
    plt.tight_layout()
    plt.show()

    print("✅ Visualización de atención completada!")
else:
    print("❌ No se pudo generar la visualización de atención")
