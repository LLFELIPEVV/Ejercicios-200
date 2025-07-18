# 🧪 Ejercicio 117/200 — LIME en TensorFlow: explicabilidad de modelo de texto simplificado
# 1️⃣ ── Importar librerías necesarias ──
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from sklearn.model_selection import train_test_split
from lime.lime_text import LimeTextExplainer

# 2️⃣ ── Dataset de ejemplo: noticias reales y falsas ──
# Simulamos 4 noticias: 2 reales (label 0) y 2 falsas (label 1)
data = {
    "text": [
        "Government launches new education policy",  # real
        "Aliens spotted near the Eiffel Tower!",  # fake
        "Health department issues new pandemic warning",  # real
        "Cure for aging discovered!",  # fake
    ],
    "label": [0, 1, 0, 1],  # 0 = Real, 1 = Fake
}
df = pd.DataFrame(data)


# 3️⃣ ── Función para limpiar texto ──
def clean_text(text):
    # Paso 1: pasar a minúsculas
    text = text.lower()
    # Paso 2: eliminar enlaces
    text = re.sub(r"http\S+|www\S+", "", text)
    # Paso 3: eliminar todo lo que no sean letras o espacios
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


# Aplicamos la limpieza a cada texto
df["text"] = df["text"].apply(clean_text)

# 4️⃣ ── Dividir en entrenamiento y prueba ──
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.5, random_state=42
)

# 5️⃣ ── Crear vectorizador de texto ──
vectorizer = TextVectorization(
    max_tokens=1000,  # máximo de palabras en el vocabulario
    output_mode="int",  # salida como enteros (índices)
    output_sequence_length=20,  # cada texto tendrá exactamente 20 tokens
)

# "Entrenar" el vectorizador con los textos
vectorizer.adapt(X_train)

# 6️⃣ ── Definir modelo simplificado ──
model = Sequential(
    [
        Embedding(input_dim=1000, output_dim=16),  # Capa de embeddings
        GlobalAveragePooling1D(),  # Promedia los embeddings
        Dense(1, activation="sigmoid"),  # Clasificación binaria (real o fake)
    ]
)

# Compilar el modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entrenar el modelo sobre los textos vectorizados
model.fit(vectorizer(X_train), y_train, epochs=10, verbose=0)

# 7️⃣ ── Función para LIME: recibe textos y devuelve probabilidades ──
class_names = ["Real", "Fake"]


def predict_proba(texts):
    """
    LIME necesita una función que reciba textos crudos
    y devuelva una matriz con la probabilidad de cada clase.
    """
    vectorized = vectorizer(tf.constant(texts))  # Vectorizamos los textos
    preds = model.predict(vectorized)  # Predicciones (1 valor por texto)
    probs = np.hstack([1 - preds, preds])  # Convertimos a [p_real, p_fake]
    return probs


# 8️⃣ ── Inicializar el explicador de texto LIME ──
explainer = LimeTextExplainer(class_names=class_names)

# 9️⃣ ── Elegir un texto de prueba a explicar ──
test_text = "Aliens spotted near Eiffel Tower"
exp = explainer.explain_instance(test_text, predict_proba, num_features=5)

# 🔟 ── Mostrar predicciones en gráfico Seaborn ──
# Extraer probabilidades del modelo
probs = predict_proba([test_text])[0]
probs_df = pd.DataFrame({"Clase": ["Real", "Fake"], "Probabilidad": probs})

# Estilo de gráfico
sns.set_theme(style="whitegrid")
sns.set_palette(palette="coolwarm")
plt.figure(figsize=(6, 4))
sns.barplot(data=probs_df, x="Clase", y="Probabilidad")
plt.title("Probabilidad predicha para el texto: «Aliens spotted...»", fontsize=12)
plt.ylim(0, 1)
plt.ylabel("Probabilidad")
plt.xlabel("Clase")
plt.tight_layout()
plt.show()

# Guardar explicabilidad LIME en archivo HTML
print("Texto evaluado:", test_text)
exp.save_to_file("lime_explanation.html")
print("✅ Explicación generada como archivo: lime_explanation.html")
