# 🧪 Ejercicio 81/200 — Regresión Logística vs SVM en Clasificación de Texto
# ------------------------------------------------------------
# Objetivo: Comparar el desempeño de regresión logística y SVM
# usando vectores de texto (CountVectorizer y TF-IDF).
# ------------------------------------------------------------
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ------------------------------------------------------------
# 📥 1. Carga y preparación del dataset
# ------------------------------------------------------------

# Leer datasets separados de noticias falsas y verdaderas
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")

# Etiquetar los datos: 0 = Fake, 1 = Real
df_fake["label"] = 0
df_true["label"] = 1

# Concatenar ambos datasets y quedarnos solo con texto y etiqueta
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df[["text", "label"]].dropna()

# Validación mínima
assert not df.isnull().values.any(), "Hay valores nulos en el dataset."
assert set(df["label"].unique()) == {0, 1}, "Las etiquetas deben ser binarias (0, 1)."

# Variables de entrada (X) y salida (y)
X = df["text"].values
y = df["label"].values

# ------------------------------------------------------------
# 🧪 2. División del conjunto de datos
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # Mantiene la proporción de clases
)

# ------------------------------------------------------------
# 🧠 3. Definición de pipelines
# ------------------------------------------------------------

# 📌 Regresión Logística con vectores de frecuencia (Bag of Words)
logistic_pipeline = Pipeline(
    [
        (
            "vectorizer",
            CountVectorizer(
                stop_words="english",  # Elimina palabras irrelevantes comunes
                max_features=10000,  # Limita el vocabulario para eficiencia
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,  # Aumentamos iteraciones para asegurar convergencia
                solver="lbfgs",  # Solución robusta para clasificación binaria
                random_state=42,
            ),
        ),
    ]
)

# 📌 SVM con vectores TF-IDF
svm_pipeline = Pipeline(
    [
        ("vectorizer", TfidfVectorizer(stop_words="english", max_features=10000)),
        ("classifier", LinearSVC(max_iter=1000, random_state=42)),
    ]
)

# ------------------------------------------------------------
# 📊 4. Entrenamiento y evaluación
# ------------------------------------------------------------


def evaluar_modelo(nombre: str, pipeline, X_train, X_test, y_train, y_test):
    """
    Entrena un pipeline y reporta métricas de evaluación.
    """
    print(f"\n=== {nombre} ===")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Informe de métricas por clase
    print("\n📈 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    # Matriz de confusión
    print("🧾 Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))


# Evaluar Regresión Logística
evaluar_modelo(
    "Regresión Logística (CountVectorizer)",
    logistic_pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
)

# Evaluar SVM
evaluar_modelo("SVM (TF-IDF)", svm_pipeline, X_train, X_test, y_train, y_test)
