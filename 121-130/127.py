# ✅ Ejercicio 127/200 — Limpieza profunda + visualización + TextVectorization con regex
# Paso 1️⃣: Importar librerías necesarias
import re  # Para limpieza de texto usando expresiones regulares
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import TextVectorization

# Paso 2️⃣: Crear un pequeño dataset simulado
data = {
    "text": [
        "Breaking: Aliens-invade (earth)!",
        "This is totally FAKE!!!",
        "NASA confirms water on Mars...",
        "Click here to win $$$",
        "Official source: Science wins.",
        "🚨 ALERT: Bananas cure cancer?!",
    ]
}

df = pd.DataFrame(data)


# Paso 3️⃣: Definir función de limpieza profunda
def limpiar(texto):
    """
    Esta función limpia el texto de forma avanzada:
    - Minúsculas
    - Elimina emojis, signos, símbolos, enlaces, dígitos y espacios duplicados
    """
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)  # eliminar enlaces
    texto = re.sub(r"\d+", "", texto)  # eliminar números
    texto = re.sub(r"[^\w\s]", " ", texto)  # quitar puntuación, dejar espacios
    texto = re.sub(r"\s+", " ", texto).strip()  # quitar espacios repetidos
    return texto


# Aplicamos limpieza a cada texto
df["clean_text"] = df["text"].apply(limpiar)

# Paso 4️⃣: Visualizar palabras más frecuentes
# Unir todos los textos en uno solo y separarlo en palabras
palabras = " ".join(df["clean_text"]).split()
frecuencia = pd.Series(palabras).value_counts().reset_index()
frecuencia.columns = ["palabra", "frecuencia"]

# Visualizar con seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.barplot(data=frecuencia.head(10), x="frecuencia", y="palabra", palette="mako")
plt.title("🔍 Palabras más frecuentes después de limpieza")
plt.xlabel("Frecuencia")
plt.ylabel("Palabra")
plt.tight_layout()
plt.show()

# Paso 5️⃣: Tokenización avanzada con expresión regular
# Aquí tokenizamos por cada palabra de 3 letras o más
vectorizador = TextVectorization(
    max_tokens=100,  # vocabulario limitado
    output_sequence_length=10,  # longitud máxima por oración
    standardize=None,  # ya limpiamos
    split="regex",  # activamos split por regex
    split_pattern=r"\b\w{3,}\b",  # solo palabras con ≥3 letras
)

# Adaptamos el vectorizador al texto limpio
vectorizador.adapt(df["clean_text"])

# Paso 6️⃣: Mostrar tokens generados
# Convertimos algunos textos a secuencias de enteros (tokens)
textos_test = df["clean_text"].iloc[:3]
tokens_generados = vectorizador(textos_test)

# Mostramos resultados
print("\n📎 Ejemplos de textos limpios y tokens generados:\n")
for i, texto in enumerate(textos_test):
    print(f"Texto limpio: {texto}")
    print(f"Tokens: {tokens_generados[i].numpy()}\n")
