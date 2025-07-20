# ✅ Ejercicio 150/200 – Predicción desde archivo .json + Validación con assert y visualización profesional
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.layers import TextVectorization
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Paso 1: Leer el archivo JSON
with open("entrada.json", "r", encoding="utf-8") as archivo:
    datos = json.load(archivo)

# Paso 2: Validar estructura
assert "noticias" in datos, "El JSON debe contener una lista bajo la clave 'noticias'"
assert isinstance(datos["noticias"], list), "'noticias' debe ser una lista"
assert len(datos["noticias"]) > 0, "La lista de noticias no puede estar vacía"

# Paso 3: Extraer textos
textos = []
ids = []
for noticia in datos["noticias"]:
    assert "contenido" in noticia, "Cada entrada debe tener clave 'contenido'"
    textos.append(noticia["contenido"])
    ids.append(noticia["id"])

# Paso 4: Vectorización básica (simula entrenado)
vectorizador = TextVectorization(output_mode="int", output_sequence_length=100)
vectorizador.adapt(textos)  # En producción, usar el mismo adaptado del entrenamiento

# Paso 5: Transformar textos
X = vectorizador(np.array(textos))

# Paso 6: Cargar modelo
modelo = load_model("modelo_noticias.h5")

# Paso 7: Predecir
predicciones = modelo.predict(X)
pred_clases = (predicciones > 0.5).astype(int).flatten()

# Validaciones
assert predicciones.shape[0] == len(textos), "Cantidad de predicciones no coincide"
assert np.all(predicciones >= 0) and np.all(predicciones <= 1), (
    "Predicciones fuera de rango"
)

# Paso 8: Guardar predicciones en CSV
df_resultado = pd.DataFrame(
    {
        "id": ids,
        "contenido": textos,
        "probabilidad_fake": predicciones.flatten(),
        "clase_predicha": pred_clases,
    }
)
df_resultado.to_csv("predicciones.csv", index=False)

# Paso 9: Métricas (ejemplo con etiquetas simuladas reales)
y_true = np.array(
    [1 if i % 2 == 0 else 0 for i in range(len(pred_clases))]
)  # Falsos y verdaderos intercalados

# Matriz de confusión
cm = confusion_matrix(y_true, pred_clases)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"],
)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("matriz_confusion.png")
plt.close()

# Curva ROC
fpr, tpr, _ = roc_curve(y_true, predicciones)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("curva_roc.png")
