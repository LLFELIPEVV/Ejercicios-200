# 🧠 Ejercicio 156/200 — Validación de Integridad del Dataset con assert y Limpieza Eficiente de Texto
import pandas as pd
import re

# Paso 1: Leer el archivo CSV original
try:
    df = pd.read_csv(
        "noticias.csv"
    )  # Asegúrate de tener este archivo con 'title' y 'label'
except FileNotFoundError:
    print("❌ Archivo 'noticias.csv' no encontrado.")
    exit()

# Paso 2: Validar columnas esperadas
assert "title" in df.columns, "❌ Falta la columna 'title'."
assert "label" in df.columns, "❌ Falta la columna 'label'."

# Paso 3: Eliminar filas duplicadas
df.drop_duplicates(subset="title", inplace=True)


# Paso 4: Función de limpieza personalizada
def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(
        r"[^a-záéíóúüñ0-9.,!?¡¿\s]", "", texto
    )  # Quitar símbolos no deseados
    texto = re.sub(r"\s+", " ", texto).strip()  # Espacios múltiples a uno solo
    return texto


# Paso 5: Aplicar limpieza al título
df["title"] = df["title"].apply(limpiar_texto)

# Paso 6: Validación adicional: asegurar que no haya textos vacíos
assert df["title"].str.strip().replace("", pd.NA).notna().all(), (
    "❌ Hay títulos vacíos tras limpieza."
)

# Paso 7: Vista previa de los primeros textos limpios
print("🧼 Muestra de noticias limpias:")
print(df[["title", "label"]].head())

# Paso 8: Guardar el DataFrame limpio a nuevo archivo
df.to_csv("noticias_limpias.csv", index=False)
print("✅ Archivo limpio guardado como 'noticias_limpias.csv'")
