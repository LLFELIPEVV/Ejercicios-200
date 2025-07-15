# 🧪 Ejercicio 93/200 — Comparación de modelos clásicos con vectores de texto
# Objetivo: Comparar dos modelos clásicos (Regresión logística y SVM)
# usando dos formas distintas de convertir texto en números (CountVectorizer y TfidfVectorizer)

# 📦 Importamos las bibliotecas necesarias
import pandas as pd  # Para manejar los datos como tablas
from sklearn.svm import LinearSVC  # Support Vector Machine lineal
from sklearn.linear_model import LogisticRegression  # Regresión logística
from sklearn.model_selection import (
    train_test_split,
)  # Para separar datos en entrenamiento y prueba
from sklearn.metrics import classification_report  # Para evaluar el modelo
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)  # Para transformar texto en números

# -----------------------------------------------------------
# 📥 1. Carga de datos: Noticias reales y falsas
# -----------------------------------------------------------
# Leemos los archivos CSV de noticias falsas y verdaderas
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna()  # Eliminamos filas vacías
df_true = pd.read_csv("Datasets/archive/True.csv").dropna()

# Creamos una nueva columna 'label' para identificar las clases
# 0 = Noticia falsa, 1 = Noticia real
df_fake["label"] = 0
df_true["label"] = 1

# Unimos ambos DataFrames en uno solo y seleccionamos solo texto y etiquetas
df = pd.concat([df_fake, df_true])[["text", "label"]]

# Separamos las noticias (X) y las etiquetas (y)
X, y = df["text"].values, df["label"].values

# -----------------------------------------------------------
# ✂️ 2. División de los datos en entrenamiento y prueba
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # 20% de los datos se usan para prueba
    random_state=42,  # Asegura que siempre tengamos la misma división
    stratify=y,  # Mantiene el mismo balance entre reales y falsas
)

# -----------------------------------------------------------
# 🧠 3. Vectorización del texto (de palabras a números)
# -----------------------------------------------------------

# CountVectorizer convierte cada palabra en su frecuencia dentro del texto
count_vec = CountVectorizer(max_features=5000, stop_words="english")

# TfidfVectorizer también cuenta palabras, pero les da menor peso si son muy comunes
tfidf_vec = TfidfVectorizer(max_features=5000, stop_words="english")

# Convertimos los textos de entrenamiento
X_train_count = count_vec.fit_transform(X_train)
X_train_tfidf = tfidf_vec.fit_transform(X_train)

# Para los datos de prueba, solo usamos "transform" para no aprender de ellos
X_test_count = count_vec.transform(X_test)
X_test_tfidf = tfidf_vec.transform(X_test)

# -----------------------------------------------------------
# 🏋️ 4. Entrenamiento de los modelos
# -----------------------------------------------------------

# Creamos una instancia de Regresión Logística (hasta 1000 iteraciones)
logistic = LogisticRegression(max_iter=1000)

# Creamos una SVM lineal (más eficiente para texto que la SVM normal)
svm = LinearSVC()

# Entrenamos cada modelo con su respectivo vector de entrada
logistic.fit(X_train_count, y_train)
svm.fit(X_train_tfidf, y_train)

# -----------------------------------------------------------
# 🧪 5. Predicción y evaluación de los modelos
# -----------------------------------------------------------

# Usamos el modelo entrenado para hacer predicciones sobre los datos de prueba
y_pred_logistic = logistic.predict(X_test_count)
y_pred_svm = svm.predict(X_test_tfidf)

# Mostramos un reporte de métricas como precisión, recall y F1-score
print("📊 Regresión Logística con CountVectorizer:\n")
print(classification_report(y_test, y_pred_logistic, target_names=["Fake", "Real"]))

print("📊 SVM con TfidfVectorizer:\n")
print(classification_report(y_test, y_pred_svm, target_names=["Fake", "Real"]))
