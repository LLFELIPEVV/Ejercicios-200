# 🧠 Ejercicio 151/200 — Organización de scripts y limpieza automatizada de texto para detección de noticias falsas
# 📂 Estructura esperada del proyecto
# fake_news_project/
# │
# ├── main.py                 # Ejecuta la limpieza
# ├── config.py               # Define rutas y parámetros del proyecto
# ├── preprocessing/
# │   └── cleaner.py          # Contiene funciones de limpieza de texto
# ├── data/
# │   ├── raw.csv             # Archivo original de entrada
# │   └── cleaned.csv         # Archivo limpio de salida

# 📁 config.py
import os

# Carpeta raíz
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de los archivos
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned.csv")

# 📁 preprocessing/cleaner.py
import re
import pandas as pd


def clean_text(text):
    """
    Limpia una cadena de texto aplicando reglas básicas.
    """
    text = text.lower()  # Minúsculas
    text = re.sub(r"http\S+", "", text)  # URLs
    text = re.sub(r"[^a-zñáéíóúü\s]", "", text)  # Elimina símbolos
    text = re.sub(r"\s+", " ", text).strip()  # Espacios múltiples
    return text


def clean_dataframe(df, text_column="text"):
    """
    Limpia un DataFrame textual:
    - Elimina duplicados
    - Aplica limpieza al texto
    - Elimina filas vacías
    """
    df = df.drop_duplicates()
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    df = df[df[text_column].str.strip() != ""]
    return df


# 📁 main.py
import pandas as pd
from config import RAW_DATA_PATH, CLEANED_DATA_PATH
from preprocessing.cleaner import clean_dataframe


def main():
    # 1. Cargar datos crudos
    df = pd.read_csv(RAW_DATA_PATH)

    # 2. Limpiar el contenido textual
    cleaned_df = clean_dataframe(df, text_column="text")

    # 3. Validación con assert
    assert cleaned_df.shape[0] > 0, "Error: ¡No quedan datos después de limpiar!"

    # 4. Guardar el nuevo archivo
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"✅ Limpieza completada. Registros finales: {cleaned_df.shape[0]}")


if __name__ == "__main__":
    main()
