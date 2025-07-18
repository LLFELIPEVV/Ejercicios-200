# 🧠 Ejercicio 123/200 – Visualización y análisis de clases desbalanceadas en un dataset textual
# Paso 1: Importar librerías necesarias
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paso 2: Crear un dataset simulado
data = {
    "text": [
        "Breaking news: something real happened!",
        "Shocking! Click here to know the truth",
        "Government confirms the report",
        "Aliens landed in Canada",
        "Study reveals health benefits of green tea",
        "Fake news: NASA hides discovery",
        "Real news: Scientists confirm theory",
        "BREAKING: hoax alert!",
        "Real: President gives new speech",
        "Hoax: cure for cancer found in bananas",
    ],
    "label": [
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
    ],
}

df = pd.DataFrame(data)


# Paso 3: Limpieza básica del texto usando expresiones regulares
def limpiar_texto(texto):
    texto = texto.lower()  # convertir a minúsculas
    texto = re.sub(r"http\S+", "", texto)  # eliminar URLs
    texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)  # eliminar símbolos y puntuación
    texto = re.sub(r"\s+", " ", texto).strip()  # quitar espacios múltiples
    return texto


df["clean_text"] = df["text"].apply(limpiar_texto)

# Paso 4: Contar y visualizar la frecuencia de clases
conteo = df["label"].value_counts()
porcentaje = df["label"].value_counts(normalize=True) * 100

print("Distribución de clases:")
print(conteo)
print("\nPorcentajes:")
print(porcentaje.round(2))

# Paso 5: Visualización con Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label", palette="viridis")
plt.title("Distribución de clases: real vs fake")
plt.xlabel("Clase")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()
