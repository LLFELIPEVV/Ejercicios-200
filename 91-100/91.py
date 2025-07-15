# 🧪 Ejercicio 91/200 — Comparación directa entre CountVectorizer y TfidfVectorizer
# Objetivo: Mostrar cómo dos técnicas diferentes convierten texto en números que pueden ser usados por algoritmos de Machine Learning.

# ----------------------------------------------------------
# 📦 1. Importación de bibliotecas necesarias
# ----------------------------------------------------------
import pandas as pd  # Para trabajar con tablas de datos (como Excel en código)
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)  # Herramientas para convertir texto a números

# ----------------------------------------------------------
# 📰 2. Definimos un pequeño conjunto de textos ("noticias")
# ----------------------------------------------------------
# Este conjunto se llama 'corpus'. Aquí simulamos 3 frases como si fueran noticias reales.
corpus = [
    "Breaking news: aliens landed on the moon.",  # Noticia 1
    "Government denies that aliens landed.",  # Noticia 2
    "Aliens might be real, says scientist.",  # Noticia 3
]

# ----------------------------------------------------------
# 🔢 3. CountVectorizer: cuenta cuántas veces aparece cada palabra
# ----------------------------------------------------------
# Esta técnica transforma el texto en una tabla que indica cuántas veces aparece cada palabra en cada noticia.
count_vec = CountVectorizer()  # Creamos el vectorizador de conteo
X_count = count_vec.fit_transform(
    corpus
)  # Aplicamos el vectorizador sobre las noticias

# Creamos una tabla (DataFrame) con los resultados para visualizar fácilmente
df_count = pd.DataFrame(
    X_count.toarray(),  # Convertimos los datos a una tabla de números
    columns=count_vec.get_feature_names_out(),  # Usamos las palabras como nombres de columnas
)

# Mostramos la tabla generada por CountVectorizer
print("🔢 CountVectorizer (frecuencias de palabras):\n")
print(df_count)

# ----------------------------------------------------------
# 📊 4. TfidfVectorizer: mide cuán importante es una palabra
# ----------------------------------------------------------
# Esta técnica también cuenta palabras, pero les da más peso si son poco comunes (es decir, si no aparecen en todas las noticias).
tfidf_vec = TfidfVectorizer()  # Creamos el vectorizador TF-IDF
X_tfidf = tfidf_vec.fit_transform(
    corpus
)  # Aplicamos el vectorizador a las mismas noticias

# Creamos una tabla con los resultados de TF-IDF
df_tfidf = pd.DataFrame(
    X_tfidf.toarray(),  # Convertimos el resultado a tabla numérica
    columns=tfidf_vec.get_feature_names_out(),  # Ponemos las palabras como encabezados
)

# Mostramos la tabla con valores TF-IDF, redondeados a 2 decimales para facilitar su lectura
print("\n📊 TfidfVectorizer (ponderación TF-IDF):\n")
print(df_tfidf.round(2))
