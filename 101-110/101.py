# 🧠 Ejercicio 101/200 — Visualización de predicciones con LIME en un modelo de detección de fake news

# 📚 Librerías necesarias
import random
import pandas as pd
import seaborn as sns

from sklearn.pipeline import make_pipeline  # Para unir preprocesamiento + modelo
from sklearn.linear_model import LogisticRegression  # Clasificador lineal eficiente
from sklearn.model_selection import (
    train_test_split,
)  # Separar datos en entrenamiento y prueba
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # Convertir texto a vectores numéricos
from lime.lime_text import LimeTextExplainer  # Librería para explicar predicciones

# 🎲 Fijamos estilo gráfico de seaborn
sns.set(style="whitegrid", font_scale=1.1)

# ============================
# 📥 Carga y etiquetado de datos
# ============================

# Cargamos noticias falsas y verdaderas
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna().sample(500, random_state=42)
df_true = pd.read_csv("Datasets/archive/True.csv").dropna().sample(500, random_state=42)

# Asignamos etiqueta 0 a fake y 1 a real
df_fake["label"] = 0
df_true["label"] = 1

# Combinamos ambos datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# Separamos texto (X) y etiquetas (y)
X = df["text"].values
y = df["label"].values

# Dividimos datos para entrenar y evaluar el modelo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ================================
# 🔠 Vectorización + Modelo
# ================================

# Creamos un vectorizador que convierte texto a vectores numéricos usando TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

# Creamos pipeline que une el vectorizador y el modelo de regresión logística
pipeline = make_pipeline(vectorizer, LogisticRegression(max_iter=300))

# Entrenamos el modelo con los datos de entrenamiento
pipeline.fit(X_train, y_train)

# ================================
# 📊 Explicación de predicción con LIME
# ================================

# Creamos explicador LIME para texto
explainer = LimeTextExplainer(class_names=["Fake", "Real"])

# Seleccionamos una noticia al azar del conjunto de prueba
i = random.randint(0, len(X_test) - 1)

print("📰 Noticia seleccionada:\n")
print(X_test[i])
print("\n🔍 LIME interpretando por qué el modelo clasificó la noticia...\n")

# Generamos explicación local con LIME (muestra 10 palabras más influyentes)
exp = explainer.explain_instance(
    X_test[i], pipeline.predict_proba, num_features=10, top_labels=1
)

# O bien (para guardar o mostrar como HTML en entornos fuera de Jupyter):
exp.save_to_file("lime_explanation.html")
