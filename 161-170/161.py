# 🧠 Ejercicio 161/200 — Organización Profesional + Limpieza Automatizada de Texto desde .csv
import re
import sys
import pandas as pd


# ---------------------------
# Función: limpiar texto
# ---------------------------
def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()  # pasar a minúsculas
    texto = re.sub(r"<.*?>", "", texto)  # eliminar HTML
    texto = re.sub(r"[^a-zA-Z0-9\s]", "", texto)  # eliminar símbolos raros
    texto = re.sub(r"\s+", " ", texto)  # reemplazar múltiples espacios por uno
    return texto.strip()


# ---------------------------
# Función: validar datos
# ---------------------------
def validar_datos(df):
    assert not df["texto"].isna().any(), "Hay valores NaN en la columna 'texto'"
    assert not df["etiqueta"].isna().any(), "Hay valores NaN en la columna 'etiqueta'"
    assert all(df["texto"].str.len() > 0), "Hay textos vacíos después de la limpieza"
    assert df["etiqueta"].isin([0, 1]).all(), "Las etiquetas deben ser 0 o 1"


# ---------------------------
# Función principal
# ---------------------------
def main(ruta_csv):
    print(f"Leyendo archivo desde: {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    print(f"Textos originales: {len(df)}")

    # Renombrar columnas si es necesario
    df.columns = [col.strip().lower() for col in df.columns]
    if "text" in df.columns:
        df.rename(columns={"text": "texto"}, inplace=True)
    if "label" in df.columns:
        df.rename(columns={"label": "etiqueta"}, inplace=True)

    # Eliminar duplicados
    df = df.drop_duplicates(subset=["texto"])

    # Limpiar textos
    df["texto"] = df["texto"].apply(limpiar_texto)

    # Validar estructura y contenido
    validar_datos(df)

    print(f"Textos después de limpieza: {len(df)}")
    print("\nEjemplo antes y después:")
    print("Original:", df.iloc[0]["texto"])
    print("Etiqueta:", df.iloc[0]["etiqueta"])

    # Guardar versión limpia (opcional)
    df.to_csv("salida_limpia.csv", index=False)
    print("\nArchivo 'salida_limpia.csv' guardado con éxito.")


# ---------------------------
# Punto de entrada
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python limpieza_texto.py archivo.csv")
        sys.exit(1)

    archivo = sys.argv[1]
    main(archivo)
