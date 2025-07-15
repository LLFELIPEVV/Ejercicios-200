# 🧪 Ejercicio 89/200 — Interpretabilidad avanzada con SHAP sobre una red densa para detección de fake news
# 🧼 Limpieza previa del entorno
import gc
import os
import shap
import pandas as pd
import tensorflow as tf

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------
# ⚙️ Configuración de CPU e hilos para uso eficiente
# ------------------------------------------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Batch reducido para equipos modestos
batch_size = 32 if num_threads <= 4 else 64 if num_threads <= 8 else 128

# ------------------------------------------------------
# 📥 Carga de datos (Fake & Real News)
# ------------------------------------------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

# Combina y filtra columnas necesarias
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

# Divide en entrenamiento y test (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------------------------------
# 🔠 Vectorización con TF-IDF (máx. 5000 palabras)
# ------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# ------------------------------------------------------
# 🧱 Definición del modelo Denso con Keras
# ------------------------------------------------------
model = Sequential(
    [
        Input(shape=(5000,)),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

# ------------------------------------------------------
# 🏋️ Entrenamiento
# ------------------------------------------------------
model.fit(
    X_train_vec,
    y_train,
    validation_data=(X_test_vec, y_test),
    epochs=3,
    batch_size=batch_size,
    verbose=1,
)

# ------------------------------------------------------
# 📊 Evaluación y métricas
# ------------------------------------------------------
y_pred = (model.predict(X_test_vec) > 0.5).astype(int).flatten()
print("\n📊 Reporte de clasificación:\n")
print(
    classification_report(
        y_test, y_pred, target_names=["Fake", "Real"], zero_division=0
    )
)

# ------------------------------------------------------
# 🔍 Interpretabilidad con SHAP (KernelExplainer)
# ------------------------------------------------------
# Seleccionamos un subconjunto pequeño para evitar sobrecarga de RAM
sample_train = X_train_vec[:50]  # Base para explicar
sample_test = X_test_vec[:1]  # Solo explicamos una muestra

# Usamos SHAP con KernelExplainer por compatibilidad
explainer = shap.KernelExplainer(model.predict, sample_train, link="logit")

# Reducimos el número de evaluaciones (por defecto usa demasiadas)
# Lo ideal es 2 * num_features + 1, pero bajamos a evitar crash de memoria
shap_values = explainer.shap_values(sample_test, nsamples=300)

# ------------------------------------------------------
# 📈 Visualización de resultados
# ------------------------------------------------------
# 1. Palabras más influyentes (positivas y negativas)
shap.summary_plot(
    shap_values,
    features=sample_test,
    feature_names=vectorizer.get_feature_names_out(),
    max_display=15,
)

# ------------------------------------------------------
# ♻️ Limpieza de recursos para evitar lentitud en VSCode
# ------------------------------------------------------
del (
    df_fake,
    df_true,
    df,
    X,
    y,
    X_train_vec,
    X_test_vec,
    shap_values,
    sample_test,
    sample_train,
)
gc.collect()
K.clear_session()
