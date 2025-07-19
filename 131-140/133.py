# 🧠 Ejercicio 133/200 — Limpieza semi-automatizada de texto con Pandas: funciones personalizadas aplicadas por lote
import re
import pandas as pd

# Simulamos una carga real
data = {
    "text": [
        "BREAKING: NASA Discovers new planet?! 😱 Visit https://nasa.gov",
        "The president said, 'We must act now.' #climatechange",
        "WIN a FREE iPhone now!!! Click 👉👉👉 http://bit.ly/scamlink",
        "Scientists develop vaccine in record time. More at https://news.com",
    ],
    "label": [0, 0, 1, 0],  # 0 = real, 1 = fake
}

df = pd.DataFrame(data)


def clean_text(text):
    text = text.lower()  # Minúsculas
    text = re.sub(r"http\S+", "", text)  # Eliminar URLs
    text = re.sub(r"[^\w\s]", "", text)  # Eliminar puntuación
    text = re.sub(r"\s+", " ", text).strip()  # Eliminar espacios extra
    return text


# Aplicar la función a cada texto con .apply()
df["cleaned_text"] = df["text"].apply(clean_text)

print("ANTES:\n", df[["text"]])
print("\nDESPUÉS:\n", df[["cleaned_text"]])
