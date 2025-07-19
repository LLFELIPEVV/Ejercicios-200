# 🧠 Ejercicio 145/200: Ensemble liviano de modelos simples con votación y validación robusta
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

# Paso 1: Cargar y preparar datos desde CSV
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")

# Etiquetado
df_fake["label"] = 0
df_true["label"] = 1

# Unir y mezclar
data = pd.concat([df_fake, df_true])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Validación de integridad
assert "text" in data.columns and "label" in data.columns
assert data["text"].notnull().all(), "Hay textos vacíos"

# Conversión de texto a características simples (número de caracteres)
data["length"] = data["text"].apply(len)

# Features y etiquetas
X = data[["length"]].values
y = data["label"].values

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# Paso 2: Definir función para crear modelos ligeros
def build_model(seed=1):
    np.random.seed(seed)
    model = Sequential(
        [
            Input(shape=(1,)),  # Solo la longitud
            Dense(8, activation="relu"),
            Dropout(0.1),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Entrenar 3 modelos con diferente semilla (para que no sean idénticos)
models = []
for i in range(3):
    m = build_model(seed=42 + i)
    m.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    models.append(m)

# Paso 3: Predicciones individuales
predictions = [model.predict(X_test) for model in models]
for pred in predictions:
    assert pred.shape == (len(X_test), 1)


# Paso 4: Votación de mayoría
def majority_vote(preds, threshold=0.5):
    # Convertimos a 0/1
    bin_preds = [np.where(p >= threshold, 1, 0) for p in preds]
    stacked = np.stack(bin_preds, axis=-1)  # (samples, models)
    majority = np.round(np.mean(stacked, axis=-1)).astype(int)
    return majority


# Predicción combinada
ensemble_preds = majority_vote(predictions)

# Paso 5: Evaluación con matriz de confusión
cm = confusion_matrix(y_test, ensemble_preds)
print("\n📊 Matriz de confusión del ensemble:")
print(cm)

# Paso 6: Curva ROC del ensemble
# Promedio de predicciones para obtener probabilidad final
ensemble_proba = np.mean([p.flatten() for p in predictions], axis=0)
fpr, tpr, _ = roc_curve(y_test, ensemble_proba)
roc_auc = auc(fpr, tpr)

# Graficar curva ROC
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curva ROC - Ensemble")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
