# 🧠 Ejercicio 108/200 — Explicabilidad con LIME sobre modelo de fake news (TensorFlow + LIME + tf.data.Dataset)
# ================================
# 📚 1. Importar librerías necesarias
# ================================
import os
import gc
import lime
import lime.lime_text
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import load_model
from keras.layers import TextVectorization

# ================================
# ⚙️ 2. Configurar entorno y matplotlib
# ================================
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(2)

# Configurar matplotlib para mostrar gráficos
plt.ion()  # Modo interactivo
plt.style.use("default")  # Estilo por defecto

# ================================
# 🧾 3. Cargar datos y preparar dataset
# ================================
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna().sample(500, random_state=42)
df_true = pd.read_csv("Datasets/archive/True.csv").dropna().sample(500, random_state=42)
df_fake["label"] = 0  # Falso
df_true["label"] = 1  # Real

# Unimos ambos dataframes
df = pd.concat([df_fake, df_true])[["text", "label"]]
X = df["text"].values
y = df["label"].values

# ================================
# 🔀 4. Separar datos
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ================================
# ✨ 5. Vectorización del texto
# ================================
vocab_size = 5000
max_len = 100

vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=max_len)
vectorizer.adapt(tf.convert_to_tensor(X_train))  # Aprende el vocabulario


# ================================
# 📦 6. Dataset con tf.data para eficiencia
# ================================
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorizer(text), label


ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
ds_test = ds_test.map(vectorize_text).batch(32).prefetch(tf.data.AUTOTUNE)

# ================================
# 📥 7. Cargar el modelo entrenado (.h5)
# ================================
model = load_model("modelo_fake_news.h5")
print("✅ Modelo cargado correctamente.")


# ================================
# 🔍 8. Aplicar LIME para explicación
# ================================
def predict_fn(texts):
    """Función auxiliar para predecir texto crudo"""
    try:
        # Vectorizar los textos
        vectorized_texts = vectorizer(texts)
        # Predecir con el modelo
        probs = model.predict(vectorized_texts, verbose=0)
        # Asegurar que sea un array 2D
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        # Retornar probabilidades para ambas clases
        return np.hstack([1 - probs, probs])
    except Exception as e:
        print(f"❌ Error en predict_fn: {e}")
        return np.array([[0.5, 0.5]] * len(texts))


# Inicializar el explicador
explainer = lime.lime_text.LimeTextExplainer(class_names=["Fake", "Real"])

# Seleccionar muestra para analizar
idx = 7
sample_text = X_test[idx]
sample_label = y_test[idx]

print(f"📰 Analizando noticia {idx}:")
print(f"🏷️ Etiqueta real: {'Real' if sample_label == 1 else 'Fake'}")
print(f"📝 Texto (primeros 200 chars): {sample_text[:200]}...")

# Generar explicación LIME
try:
    print("🔍 Generando explicación LIME...")
    exp = explainer.explain_instance(
        sample_text,
        predict_fn,
        num_features=10,
        num_samples=1000,  # Reducir si es muy lento
    )

    # Obtener los pesos de las palabras
    weights = dict(exp.as_list())
    print(f"✅ Explicación generada. Palabras analizadas: {len(weights)}")

    # ================================
    # 📊 9. Visualización mejorada
    # ================================
    # Crear figura con tamaño adecuado
    plt.figure(figsize=(12, 8))

    # Separar palabras y pesos
    words = list(weights.keys())
    values = list(weights.values())

    # Crear colores: rojo para fake, azul para real
    colors = ["red" if v < 0 else "blue" for v in values]

    # Crear el gráfico de barras horizontal
    bars = plt.barh(words, values, color=colors, alpha=0.7)

    # Personalizar el gráfico
    plt.title(
        f"🔍 Palabras más influyentes según LIME\n(Predicción: {'Real' if np.mean(values) > 0 else 'Fake'})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Contribución a la predicción", fontsize=12)
    plt.ylabel("Palabras", fontsize=12)

    # Agregar línea vertical en x=0
    plt.axvline(x=0, color="black", linestyle="--", alpha=0.5)

    # Agregar leyenda
    plt.text(
        0.02,
        0.98,
        "🔴 Rojo: Contribuye a 'Fake'\n🔵 Azul: Contribuye a 'Real'",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Ajustar layout
    plt.tight_layout()

    # Guardar y mostrar
    plt.savefig("lime_barplot.png", dpi=300, bbox_inches="tight")
    print("💾 Gráfico guardado como 'lime_barplot.png'")

    # Mostrar el gráfico
    plt.show()

    # Forzar la visualización
    plt.draw()
    plt.pause(0.001)

    # ================================
    # 📋 10. Resumen de resultados
    # ================================
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE ANÁLISIS LIME")
    print("=" * 50)
    print(f"🏷️ Etiqueta real: {'Real' if sample_label == 1 else 'Fake'}")

    # Predicción del modelo
    sample_pred = predict_fn([sample_text])[0]
    predicted_class = "Real" if sample_pred[1] > sample_pred[0] else "Fake"
    confidence = max(sample_pred[0], sample_pred[1])

    print(f"🤖 Predicción del modelo: {predicted_class} (confianza: {confidence:.2%})")

    print("\n🔍 Top 5 palabras que contribuyen a 'Fake':")
    fake_words = sorted(
        [(w, v) for w, v in weights.items() if v < 0], key=lambda x: x[1]
    )
    for word, weight in fake_words[:5]:
        print(f"   • {word}: {weight:.3f}")

    print("\n🔍 Top 5 palabras que contribuyen a 'Real':")
    real_words = sorted(
        [(w, v) for w, v in weights.items() if v > 0], key=lambda x: x[1], reverse=True
    )
    for word, weight in real_words[:5]:
        print(f"   • {word}: {weight:.3f}")

    # También generar el HTML de LIME (opcional)
    print("\n💡 Generando visualización HTML de LIME...")
    exp.save_to_file("lime_explanation.html")
    print("💾 Explicación HTML guardada como 'lime_explanation.html'")

except Exception as e:
    print(f"❌ Error al generar explicación LIME: {e}")
    print("🔧 Verifica que el modelo y los datos estén cargados correctamente")

# ================================
# ♻️ 11. Limpieza de memoria
# ================================
print("\n🧹 Limpiando memoria...")
K.clear_session()
gc.collect()
print("✅ Análisis completado!")

# Mantener la ventana abierta en algunos entornos
try:
    input("Presiona Enter para continuar...")
except Exception as _:
    pass
