# 🧪 Ejercicio 92/200 — Visualización comparativa entre CountVectorizer y TfidfVectorizer con dataset real

# -----------------------------------------
# 📦 Importamos las librerías necesarias
# -----------------------------------------
import pandas as pd  # Para manejo de datos tabulares
import matplotlib.pyplot as plt  # Para crear gráficos
import seaborn as sns  # Para crear mapas de calor visualmente atractivos

# Herramientas de scikit-learn para convertir texto en números
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# -----------------------------------------
# 📥 1. Cargamos ejemplos reales de noticias falsas y verdaderas
# -----------------------------------------
# Cargamos 3 noticias falsas de manera aleatoria (sin valores vacíos)
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna().sample(3, random_state=42)

# Cargamos 3 noticias verdaderas de manera aleatoria
df_true = pd.read_csv("Datasets/archive/True.csv").dropna().sample(3, random_state=42)

# Unimos ambos grupos en un solo conjunto de datos (6 noticias)
df = pd.concat([df_fake, df_true])

# Extraemos únicamente el texto de las noticias para su análisis
texts = df["text"].values[:6]
# -----------------------------------------
# 🔢 2. Representamos el texto con CountVectorizer
# -----------------------------------------

# CountVectorizer cuenta cuántas veces aparece cada palabra en cada documento
# Limita el vocabulario a las 15 palabras más frecuentes y elimina palabras vacías como "the", "and", etc.
count_vectorizer = CountVectorizer(max_features=15, stop_words="english")

# Transformamos los textos a una matriz de frecuencias
X_count = count_vectorizer.fit_transform(texts)
# -----------------------------------------
# 📊 3. Representamos el texto con TfidfVectorizer
# -----------------------------------------

# TfidfVectorizer también convierte palabras a números, pero ajusta su importancia
# Penaliza palabras que aparecen en muchos documentos (como "news") y destaca las más informativas
tfidf_vectorizer = TfidfVectorizer(max_features=15, stop_words="english")

# Aplicamos la transformación TF-IDF a los textos
X_tfidf = tfidf_vectorizer.fit_transform(texts)
# -----------------------------------------
# 🧾 4. Convertimos los resultados a tablas (DataFrames) para visualizar
# -----------------------------------------

# Convertimos la matriz de frecuencias (CountVectorizer) a un DataFrame de Pandas
df_count = pd.DataFrame(
    X_count.toarray(),  # Convertimos la matriz dispersa a una tabla completa de números
    columns=count_vectorizer.get_feature_names_out(),  # Usamos las palabras como nombres de columnas
)

# Lo mismo para la matriz TF-IDF
df_tfidf = pd.DataFrame(
    X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
)
# -----------------------------------------
# 🔥 5. Visualizamos CountVectorizer como mapa de calor
# -----------------------------------------

# Creamos un gráfico de calor que muestra cuántas veces aparece cada palabra en cada noticia
plt.figure(figsize=(10, 5))
sns.heatmap(df_count, annot=True, cmap="Blues", cbar=False)
plt.title("CountVectorizer: Frecuencia de palabras")
plt.xlabel("Palabra")
plt.ylabel("Documento (Noticia)")
plt.show()
# -----------------------------------------
# 🎯 6. Visualizamos TfidfVectorizer como mapa de calor
# -----------------------------------------

# Mismo tipo de gráfico, pero esta vez muestra la "importancia" de cada palabra (ajustada por su frecuencia global)
plt.figure(figsize=(10, 5))
sns.heatmap(df_tfidf, annot=True, cmap="Purples", cbar=False)
plt.title("TfidfVectorizer: Importancia TF-IDF")
plt.xlabel("Palabra")
plt.ylabel("Documento (Noticia)")
plt.show()
