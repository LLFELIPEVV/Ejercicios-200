# ✅ Ejercicio 60/200 — Preprocesamiento profesional de texto con TextVectorization en Keras
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization

# 🗂️ Cargar y etiquetar datos
fake = pd.read_csv(r"Datasets/archive/Fake.csv")
true = pd.read_csv(r"Datasets/archive/True.csv")

fake["label"] = 0  # Noticias falsas
true["label"] = 1  # Noticias reales (corregido: tenías "true['true'] = 1")

# 🔗 Unificar datasets y limpiar datos nulos
df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]].dropna()

# 🧾 Separar texto y etiquetas
X = df["text"]
y = df["label"]

# 🔀 Dividir en conjunto de entrenamiento y prueba (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔤 Crear capa de vectorización de texto
vectorizador = TextVectorization(
    max_tokens=10000,  # Limita el vocabulario a las 10k palabras más frecuentes
    output_mode="int",  # Convierte texto en secuencias de enteros
    output_sequence_length=300,  # Longitud fija para cada secuencia
)

# 🧠 Entrenar el vectorizador con los textos de entrenamiento
vectorizador.adapt(X_train.values)

# 🔁 Vectorizar los textos
X_train_vectorizado = vectorizador(X_train)
X_test_vectorizado = vectorizador(X_test)

# 🔍 Mostrar ejemplo de transformación
print("📝 Texto original:")
print(X_train.iloc[0])

print("\n🔢 Vectorizado (tokens):")
print(
    X_train_vectorizado[0].numpy()
)  # .numpy() para ver el contenido si estás fuera de un modelo
