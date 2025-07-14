# 🧪 Ejercicio 85/200 — Explicabilidad básica con LIME en modelo Keras para Fake News
import os
import gc
import psutil
import lime
import numpy as np
import pandas as pd
import lime.lime_text
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
    Dense,
    Dropout,
    Add,
    GlobalAveragePooling1D,
    TextVectorization,
)
from keras.optimizers import Adam

# --------------------------------------------------
# ⚙️ Configuración eficiente del entorno
# --------------------------------------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

batch_size = 32 if num_threads <= 4 else 64 if num_threads <= 8 else 128

# --------------------------------------------------
# 📥 Carga y limpieza del dataset
# --------------------------------------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------------------------------
# 🧹 Preprocesamiento clásico con TF-IDF
# --------------------------------------------------
max_tokens = 5000
sequence_length = 100

vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize="lower_and_strip_punctuation",
)

# Aprender vocabulario
vectorizer.adapt(X_train)

# --------------------------------------------------
# 🔄 Dataset eficiente con vectorización incluida
# --------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=512)
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
# 🧠 Definición de un bloque Transformer simplificado
# --------------------------------------------------
def transformer_block(
    x, num_heads=2, key_dim=32, ff_dim=32, dropout_rate=0.1, block_id=0
):
    attn = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"mha_{block_id}"
    )(x, x)
    x = Add(name=f"add_attn_{block_id}")([x, attn])
    x = LayerNormalization(name=f"ln_attn_{block_id}")(x)

    ff = Dense(ff_dim, activation="relu", name=f"ffn_{block_id}")(x)
    ff = Dropout(dropout_rate, name=f"drop_ffn_{block_id}")(ff)
    x = Add(name=f"add_ffn_{block_id}")([x, ff])
    x = LayerNormalization(name=f"ln_ffn_{block_id}")(x)
    return x


# --------------------------------------------------
# 🧱 Construcción del modelo Transformer (liviano)
# --------------------------------------------------
input_layer = Input(shape=(100,), name="input_tokens")
x = Embedding(input_dim=5000, output_dim=32, name="embedding")(input_layer)

# Apilamos 3 bloques Transformer
for i in range(3):
    x = transformer_block(x, block_id=i)

x = GlobalAveragePooling1D(name="avg_pool")(x)
x = Dropout(0.1, name="final_dropout")(x)
output = Dense(1, activation="sigmoid", name="output")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# --------------------------------------------------
# 🏋️ Entrenamiento del modelo
# --------------------------------------------------
model.fit(train_ds, validation_data=test_ds, epochs=3, verbose=1)

# --------------------------------------------------
# 📊 Evaluación y métricas
# --------------------------------------------------
y_pred = model.predict(test_ds).flatten()
y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n📈 Reporte de clasificación:")
print(classification_report(y_true, y_pred_labels, zero_division=0))

# --------------------------------------------------
# 🧠 Aplicación de LIME para explicabilidad
# --------------------------------------------------
class_names = ["Fake", "Real"]


# LIME requiere un pipeline que acepte texto crudo y devuelva probabilidades para ambas clases
def lime_predict(texts):
    """
    Función de predicción para LIME que devuelve probabilidades para ambas clases
    """
    vectorized = vectorizer(tf.constant(texts))  # Vectorizamos con Keras
    preds = model.predict(vectorized)  # Predecimos con el modelo Keras

    # Convertir probabilidades de clasificación binaria a formato de dos clases
    # preds contiene P(clase=1), necesitamos [P(clase=0), P(clase=1)]
    prob_class_1 = preds.flatten()
    prob_class_0 = 1 - prob_class_1

    # Crear array de probabilidades para ambas clases
    return np.column_stack([prob_class_0, prob_class_1])


# Instanciamos el explicador de texto
explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)

# Seleccionamos un ejemplo representativo
i = 42
sample_text = X_test[i]
print(
    f"\n📰 Texto seleccionado: {sample_text[:200]}..."
)  # Mostrar solo los primeros 200 caracteres

# Obtener predicción para mostrar
sample_pred = lime_predict([sample_text])
predicted_class = class_names[np.argmax(sample_pred[0])]
confidence = np.max(sample_pred[0])

print(f"🔍 Predicción del modelo: {predicted_class} (confianza: {confidence:.3f})")

# Generamos explicación
print("\n🔍 Generando explicación con LIME...")
explanation = explainer.explain_instance(
    sample_text,
    lime_predict,
    num_features=10,
    num_samples=1000,  # Número de muestras para LIME
)

# Mostrar explicación en formato texto
print("\n📊 Explicación LIME:")
print("=" * 50)
for feature, weight in explanation.as_list():
    direction = "REAL" if weight > 0 else "FAKE"
    print(f"'{feature}' -> {direction} (peso: {weight:.3f})")

# Mostrar explicación en notebook si está disponible
try:
    explanation.show_in_notebook(text=sample_text)
except Exception as _:
    print("\n💡 Para visualización interactiva, ejecuta en Jupyter notebook")

# Guardar explicación como HTML
try:
    explanation.save_to_file("lime_explanation.html")
    print("\n💾 Explicación guardada en 'lime_explanation.html'")
except Exception as e:
    print(f"\n❌ Error al guardar explicación: {e}")

# --------------------------------------------------
# 🧹 Limpieza de memoria post-ejercicio
# --------------------------------------------------
del (
    df_fake,
    df_true,
    df,
    X_train,
    X_test,
    y_train,
    y_test,
    train_ds,
    test_ds,
    y_pred,
    y_pred_labels,
    y_true,
    explainer,
    explanation,
    sample_text,
)

gc.collect()
print(f"\n✅ Memoria liberada. Uso actual: {psutil.virtual_memory().percent:.1f}%")
