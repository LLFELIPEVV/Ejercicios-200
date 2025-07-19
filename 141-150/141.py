# ✅ Ejercicio 141/200 – Visualización Profesional + Evaluación desde Input Real
# ✅ Paso 0: Importación de librerías necesarias
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

from keras.models import load_model
from keras.layers import TextVectorization

# ✅ Paso 1: Cargar el modelo previamente entrenado (.h5)
model = load_model("modelo_fake_news.h5")

# ✅ Paso 2: Cargar un archivo .csv con datos reales
df_fake = pd.read_csv(r"Datasets\archive\Fake.csv")
df_true = pd.read_csv(r"Datasets\archive\True.csv")

# Etiquetamos las noticias: 0 = fake, 1 = real
df_fake["label"] = 0
df_true["label"] = 1

# Unimos ambos y mezclamos aleatoriamente
df = (
    pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
)

tokenizer = TextVectorization(max_tokens=1000, output_sequence_length=200)
tokenizer.adapt(df["text"].values)

# Guardar tokenizador como objeto serializado
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Seleccionamos un texto de prueba (simula entrada real)
texto = df["text"].iloc[0]
etiqueta_real = df["label"].iloc[0]  # Obtenemos también su etiqueta real

# ✅ Paso 3: Cargar el tokenizador previamente guardado (.pickle)
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# ✅ Paso 4: Preprocesar el texto como en el entrenamiento
# Convertimos el texto en secuencia de enteros
# Rellenamos/pad la secuencia para que tenga la misma longitud usada en el modelo
entrada = tokenizer([texto])

# ✅ Paso 5: Realizamos la predicción
prediccion = model.predict(entrada, verbose=0)[0][0]

# Mostramos la probabilidad estimada de que sea noticia falsa
print(f"\n🧪 Probabilidad de 'Fake News': {prediccion:.4f}")

# ✅ Paso 6: Determinamos la clase final
pred_clase = int(prediccion > 0.5)

# ✅ Paso 7: Visualizamos la matriz de confusión
cm = confusion_matrix([etiqueta_real], [pred_clase])

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
plt.ylabel("Etiqueta Real")
plt.show()

# ✅ Paso 8: Visualizamos la curva ROC
fpr, tpr, _ = roc_curve([etiqueta_real], [prediccion])
plt.plot(fpr, tpr, label="ROC curve (AUC = {:.2f})".format(auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.show()

# ✅ Paso 9: Visualizamos la curva de Precisión-Recall
precision, recall, _ = precision_recall_curve([etiqueta_real], [prediccion])
plt.plot(recall, precision, label="Curva Precisión-Recall")
plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.title("Curva Precisión-Recall")
plt.legend()
plt.show()
