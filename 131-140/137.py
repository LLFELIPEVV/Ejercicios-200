# ✅ Ejercicio 137/200: Ensemble Voting con modelos ligeros previamente entrenados
# Objetivo: Combinar predicciones de modelos pequeños para mejorar la robustez
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mode
from keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Paso 1: Cargar tres modelos ligeros previamente entrenados (mismo vectorizador)
model1 = load_model("modelo_1.h5")
model2 = load_model("modelo_2.h5")
model3 = load_model("modelo_3.h5")

# Paso 2: Cargar dataset crudo
df_fake = pd.read_csv(r"Datasets\archive\Fake.csv")
df_true = pd.read_csv(r"Datasets\archive\True.csv")

# Etiquetar
df_fake["label"] = 0
df_true["label"] = 1

# Unir y mezclar
data = (
    pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
)
X_raw = data["text"].astype(str).values
y = data["label"].values

# Paso 3: Vectorización liviana (simulando el preprocesamiento usado durante el entrenamiento)
# ⚠️ IMPORTANTE: Asegúrate de usar los mismos parámetros que usaste para entrenar los modelos.
vectorizer = load_model("vectorizer_layer.keras")

# Ahora sí puedes transformar el texto
X_processed = vectorizer(X_raw)

# Paso 4: Predicciones de los modelos
pred1 = np.round(model1.predict(X_processed, verbose=0))
pred2 = np.round(model2.predict(X_processed, verbose=0))
pred3 = np.round(model3.predict(X_processed, verbose=0))

# Paso 5: Votación por mayoría
pred_ensemble = mode(
    np.concatenate([pred1, pred2, pred3], axis=1), axis=1
).mode.flatten()

# Paso 6: Matriz de confusión
cm = confusion_matrix(y, pred_ensemble)
ConfusionMatrixDisplay(cm).plot()
plt.title("Matriz de Confusión - Ensemble")
plt.show()

# Paso 7: Curva ROC (usamos promedio de probabilidades)
avg_probs = (pred1 + pred2 + pred3) / 3
fpr, tpr, _ = roc_curve(y, avg_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Curva ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC - Ensemble Voting")
plt.legend(loc="lower right")
plt.show()
