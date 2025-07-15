# 🧪 Ejercicio 94/200 — Explicabilidad con LIME para detección de fake news

# 📦 Importamos librerías necesarias
import pandas as pd
import lime
import lime.lime_text
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------
# 📥 Paso 1: Cargar y preparar los datos
# -------------------------------------------

# Cargamos noticias falsas y verdaderas desde archivos CSV
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna()
df_true = pd.read_csv("Datasets/archive/True.csv").dropna()

# Añadimos la columna 'label': 0 para noticias falsas, 1 para reales
df_fake["label"] = 0
df_true["label"] = 1

# Unimos ambos conjuntos en uno solo y tomamos una muestra pequeña por rendimiento
df = pd.concat([df_fake, df_true])[["text", "label"]].sample(1000, random_state=42)

# Guardamos los textos y las etiquetas en variables separadas
X = df["text"].values  # Textos (noticias)
y = df["label"].values  # Etiquetas: 0 o 1

# Dividimos el conjunto en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------------------
# 🏗️ Paso 2: Crear modelo clásico con pipeline
# -------------------------------------------

# Usamos TF-IDF para transformar el texto en números útiles para el modelo
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

# Creamos el modelo de regresión logística
model = LogisticRegression(max_iter=1000, verbose=0)

# Creamos un pipeline que primero transforma el texto y luego entrena el modelo
pipeline = make_pipeline(vectorizer, model)

# Entrenamos el modelo con los datos
pipeline.fit(X_train, y_train)

# -------------------------------------------
# 🔍 Paso 3: Usar LIME para explicar una predicción
# -------------------------------------------

# LIME necesita saber los nombres de las clases
explainer = lime.lime_text.LimeTextExplainer(class_names=["Fake", "Real"])

# Elegimos un texto cualquiera del conjunto de prueba para explicarlo
idx = 10
sample_text = X_test[idx]

# Generamos la explicación para ese texto
explanation = explainer.explain_instance(
    sample_text,  # Texto que queremos analizar
    pipeline.predict_proba,  # Método para obtener probabilidades
    num_features=10,  # Número de palabras importantes que se mostrarán
)

# Mostramos el texto original que fue analizado
print("\nTexto analizado:\n")
print(sample_text)

# -------------------------------------------
# 📊 Paso 4: Visualizar la explicación con Seaborn
# -------------------------------------------

# Obtenemos las palabras y sus pesos desde LIME
exp_data = explanation.as_list()  # [(palabra, peso), ...]
words, weights = zip(*exp_data)

# Creamos un gráfico de barras con Seaborn
plt.figure(figsize=(10, 5))
sns.barplot(x=weights, y=words, palette="coolwarm")

plt.title("Palabras que más influyeron en la predicción")
plt.xlabel("Influencia en la decisión del modelo")
plt.ylabel("Palabra")
plt.axvline(0, color="black", linewidth=1)  # Línea central
plt.tight_layout()
plt.show()
