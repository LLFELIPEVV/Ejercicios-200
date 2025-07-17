# 🧪 Ejercicio 111/200 – Aplicando LIME correctamente sobre un modelo optimizado para detectar noticias falsas
# ================================================
# 🧠 LIBRERÍAS NECESARIAS
# ================================================
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    Embedding,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    TextVectorization,
)

# ================================================
# 1. CREAR UN PEQUEÑO DATASET DE NOTICIAS
# ================================================
# Las noticias tienen una etiqueta: 1 = Falsa, 0 = Real
data = {
    "text": [
        "The president was replaced by an alien",  # Falsa
        "NASA confirms water on Mars",  # Real
        "Vaccines cause autism, experts say",  # Falsa
        "Elections were fair and transparent",  # Real
        "5G towers spread COVID-19",  # Falsa
        "The economy is recovering well",  # Real
    ],
    "label": [1, 0, 1, 0, 1, 0],
}
df = pd.DataFrame(data)

# ================================================
# 2. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"].values,
    np.array(df["label"]).astype("float32"),
    test_size=0.2,
    random_state=42,
)

# ================================================
# 3. TOKENIZACIÓN DEL TEXTO
# ================================================
# Convirtiendo el texto a secuencias numéricas para entrenar el modelo
max_tokens = 1000  # Tamaño del vocabulario limitado
sequence_length = 30  # Longitud de secuencia fija

vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize="lower_and_strip_punctuation",  # Limpieza automática
)
vectorizer.adapt(X_train)  # Aprende del texto

# ================================================
# 4. CREAR MODELO NEURONAL OPTIMIZADO
# ================================================
model = Sequential(
    [
        vectorizer,  # Preprocesamiento textual
        Embedding(input_dim=max_tokens, output_dim=16),  # Capa de representación
        GlobalAveragePooling1D(),  # Promedia vectores de palabras
        Dropout(0.4),  # Previene sobreajuste
        Dense(8, activation="relu"),  # Capa intermedia
        Dense(1, activation="sigmoid"),  # Salida binaria (0 = real, 1 = falsa)
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Callbacks para evitar entrenar de más
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=2)

# ================================================
# 5. ENTRENAMIENTO DEL MODELO
# ================================================
hist = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ================================================
# 6. VISUALIZAR ENTRENAMIENTO CON SEABORN
# ================================================
# Convertimos el historial en un DataFrame
history_df = pd.DataFrame(hist.history)

# Graficamos precisión y pérdida
plt.figure(figsize=(12, 5))

# Gráfico de precisión
plt.subplot(1, 2, 1)
sns.lineplot(data=history_df[["accuracy", "val_accuracy"]])
plt.title("Precisión del modelo")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend(["Entrenamiento", "Validación"])

# Gráfico de pérdida
plt.subplot(1, 2, 2)
sns.lineplot(data=history_df[["loss", "val_loss"]])
plt.title("Pérdida del modelo")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend(["Entrenamiento", "Validación"])

plt.tight_layout()
plt.show()

# ================================================
# 7. EXPORTAR MODELO ENTRENADO
# ================================================
model.save("modelo_fake_news.h5")

# ================================================
# 8. FUNCIÓN NECESARIA PARA QUE LIME FUNCIONE
# ================================================
class_names = ["Real", "Fake"]


def predict_proba(texts):
    """
    Función auxiliar que convierte un texto en probabilidades de clase.
    LIME requiere una matriz 2D con [Prob. Real, Prob. Falsa]
    """
    texts_tensor = tf.convert_to_tensor(texts)
    preds = model.predict(texts_tensor)
    return np.hstack([1 - preds, preds])


# ================================================
# 9. APLICAR LIME PARA EXPLICAR UNA PREDICCIÓN
# ================================================
explainer = LimeTextExplainer(class_names=class_names)

# Seleccionamos un texto del conjunto de prueba
texto_a_explicar = X_test[0]
print("\n📝 Texto a explicar:", texto_a_explicar)

# Creamos la explicación
exp = explainer.explain_instance(texto_a_explicar, predict_proba, num_features=6)

# Mostramos las palabras más importantes según LIME
print("\n📊 Importancia de palabras según LIME:\n", exp.as_list())
