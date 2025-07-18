# 🧠 Ejercicio 129/200 — Visualización de Clases Desbalanceadas + Tokenización Subpalabra con TextVectorization
import re
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import TextVectorization

# 1. Dataset simulado
texts = [
    "The government has announced a new policy for the economy.",
    "BREAKING: Aliens land in Brazil and heal cancer with crystals.",
    "Elections will be held in October, says the president.",
    "Vaccines cause autism, say fake experts with no evidence.",
    "Local farmers protest against tax increase in the region.",
    "COVID-19 is a hoax, according to misleading Facebook posts.",
]

labels = [0, 1, 0, 1, 0, 1]  # 0 = real, 1 = fake


# 2. Visualización de frecuencia de clases
def visualizar_distribucion_clases(labels):
    sns.set_theme(style="whitegrid")
    sns.countplot(x=labels, palette="pastel")
    plt.title("Distribución de clases (0=real, 1=fake)")
    plt.xlabel("Clase")
    plt.ylabel("Frecuencia")
    plt.show()


visualizar_distribucion_clases(labels)


# 3. Limpieza básica con expresiones regulares
def limpiar_texto(texto):
    texto = texto.lower()  # minúsculas
    texto = re.sub(r"http\S+", "", texto)  # eliminar links
    texto = re.sub(r"[^a-z\s]", "", texto)  # eliminar puntuación
    texto = re.sub(r"\s+", " ", texto)  # espacios múltiples
    return texto.strip()


# Aplicamos limpieza a cada texto
textos_limpios = [limpiar_texto(t) for t in texts]

# 4. Tokenización con TextVectorization usando n-gramas
vectorizador = TextVectorization(
    max_tokens=1000,  # pequeño vocabulario
    output_mode="int",
    output_sequence_length=20,
    ngrams=2,  # n-gramas para capturar combinaciones de subpalabras
)

# Adaptamos al vocabulario
vectorizador.adapt(textos_limpios)

# 5. Mostrar vocabulario aprendido
vocabulario = vectorizador.get_vocabulary()
print("\n📚 Vocabulario aprendido (primeros 20 tokens):")
print(vocabulario[:20])

# 6. Ejemplo de transformación
print("\n🔠 Texto tokenizado de ejemplo:")
ejemplo = textos_limpios[1]
print("Original:", ejemplo)
print("Tokenizado:", vectorizador(tf.constant([ejemplo])).numpy())
