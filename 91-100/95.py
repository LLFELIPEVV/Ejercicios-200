# 🧪 Ejercicio 95/200 — Interpretabilidad comparada: LIME vs SHAP en regresión logística con TF-IDF

# 🔧 Importamos las librerías necesarias
import shap
import lime
import lime.lime_text
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 🗂️ Cargamos noticias falsas y verdaderas desde archivos CSV
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna()
df_true = pd.read_csv("Datasets/archive/True.csv").dropna()

# 📌 Asignamos etiquetas: 0 = Fake, 1 = Real
df_fake["label"] = 0
df_true["label"] = 1

# 🔀 Unimos ambos datasets y tomamos una muestra pequeña para que tu PC no se sature
df = pd.concat([df_fake, df_true])[["text", "label"]].sample(1000, random_state=42)

# 🎯 X = textos de noticias, y = etiquetas (Fake o Real)
X = df["text"].values
y = df["label"].values

# ✂️ Dividimos en entrenamiento y prueba (80% para entrenar, 20% para evaluar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✏️ Convertimos texto en vectores usando TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

# 🧮 Modelo de regresión logística
model = LogisticRegression(max_iter=1000)

# 🔗 Creamos un pipeline que combina el vectorizador y el modelo
pipeline = make_pipeline(vectorizer, model)

# 🏋️ Entrenamos el modelo con los textos de entrenamiento
pipeline.fit(X_train, y_train)

# -----------------------------
# 🔍 Interpretabilidad con LIME
# -----------------------------

# Instanciamos el explicador LIME (trabaja con texto)
lime_explainer = lime.lime_text.LimeTextExplainer(class_names=["Fake", "Real"])

# Elegimos un texto de prueba para explicar
idx = 8
text_instance = X_test[idx]

# LIME analiza la importancia de cada palabra en la predicción
lime_exp = lime_explainer.explain_instance(
    text_instance, pipeline.predict_proba, num_features=10
)

# Mostramos el texto que analizamos
print("\n📰 Texto analizado (LIME):\n", text_instance)

# Extraemos las palabras más influyentes y sus puntuaciones
lime_weights = lime_exp.as_list()
words, scores = zip(*lime_weights)

# 📊 Visualización con seaborn: qué palabras influenciaron más al modelo
plt.figure(figsize=(10, 5))
# Fix 1: Asignar hue correctamente para evitar el warning
colors = ["red" if score < 0 else "blue" for score in scores]
sns.barplot(x=list(scores), y=list(words), palette=colors)
plt.title("LIME: Palabras más influyentes en la predicción")
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Influencia en la predicción")
plt.ylabel("Palabra")
plt.tight_layout()
plt.show()

# -----------------------------
# 🔍 Interpretabilidad con SHAP
# -----------------------------

# ⚠️ SHAP necesita trabajar directamente con los vectores, no con texto crudo
X_train_vec = vectorizer.transform(X_train[:100])  # solo 100 muestras para eficiencia
X_test_vec = vectorizer.transform(X_test[:1])  # solo una muestra

# Creamos un explicador SHAP (usamos KernelExplainer porque el modelo es lineal)
shap_explainer = shap.KernelExplainer(model.predict_proba, X_train_vec)

# Obtenemos las explicaciones SHAP (shap_values)
shap_values = shap_explainer.shap_values(X_test_vec, nsamples=100)

# Fix 2: Verificar la forma de shap_values antes de usar
print(f"Shape of shap_values: {np.array(shap_values).shape}")
print(f"Type of shap_values: {type(shap_values)}")

# Fix 3: Manejar diferentes formatos de shap_values
if isinstance(shap_values, list) and len(shap_values) == 2:
    # Caso normal: shap_values es una lista con 2 elementos [clase_0, clase_1]
    shap_values_to_plot = shap_values[1]  # clase "Real"
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
    # Caso alternativo: shap_values es un array 2D
    shap_values_to_plot = shap_values
else:
    # Caso de respaldo: usar el primer elemento
    shap_values_to_plot = (
        shap_values[0] if isinstance(shap_values, list) else shap_values
    )

# Visualizamos las 10 palabras más relevantes usando SHAP
try:
    shap.summary_plot(
        shap_values_to_plot,
        features=X_test_vec.toarray(),
        feature_names=vectorizer.get_feature_names_out(),
        max_display=10,
    )
except Exception as e:
    print(f"Error en SHAP summary_plot: {e}")

    # Alternativa: crear visualización manual
    print("\n📊 Creando visualización SHAP manual...")

    # Obtener los valores SHAP y nombres de características
    shap_vals = (
        shap_values_to_plot[0] if shap_values_to_plot.ndim > 1 else shap_values_to_plot
    )
    feature_names = vectorizer.get_feature_names_out()

    # Obtener índices de características con valores SHAP no cero
    non_zero_indices = np.nonzero(shap_vals)[0]

    if len(non_zero_indices) > 0:
        # Ordenar por valor absoluto de SHAP
        sorted_indices = non_zero_indices[
            np.argsort(np.abs(shap_vals[non_zero_indices]))
        ][-10:]

        # Crear gráfico manual
        plt.figure(figsize=(10, 6))
        shap_scores = shap_vals[sorted_indices]
        feature_names_selected = feature_names[sorted_indices]

        colors = ["red" if score < 0 else "blue" for score in shap_scores]
        plt.barh(range(len(shap_scores)), shap_scores, color=colors)
        plt.yticks(range(len(shap_scores)), feature_names_selected)
        plt.xlabel("Valor SHAP")
        plt.title("SHAP: Top 10 características más influyentes")
        plt.axvline(0, color="black", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("No se encontraron características con valores SHAP significativos")

# 📊 Información adicional sobre la predicción
prediction = pipeline.predict([text_instance])[0]
prediction_proba = pipeline.predict_proba([text_instance])[0]

print("\n🎯 Predicción del modelo:")
print(f"Clase predicha: {'Real' if prediction == 1 else 'Fake'}")
print(f"Probabilidad Fake: {prediction_proba[0]:.3f}")
print(f"Probabilidad Real: {prediction_proba[1]:.3f}")
