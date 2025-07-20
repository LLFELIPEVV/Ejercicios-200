# 🧠 Ejercicio 167/200 — Limpieza Automática de Texto con Validación y Guardado
import re
import sys
import html
import pandas as pd

from emoji import replace_emoji


# ---------------------------
# Función: Validación inicial
# ---------------------------
def validar_dataframe(df):
    assert "texto" in df.columns, "El archivo debe tener una columna llamada 'texto'"
    assert df["texto"].notnull().all(), "Existen textos vacíos o nulos"
    assert df.shape[0] > 0, "El archivo está vacío"
    print(f"[INFO] Registros iniciales: {df.shape[0]}")


# ---------------------------
# Función: Limpieza de texto
# ---------------------------
def limpiar_texto(texto):
    texto = str(texto)  # Asegura que sea string
    texto = html.unescape(texto)  # Decodifica entidades HTML (&amp;, &quot;)
    texto = texto.lower()  # Convierte a minúsculas
    texto = re.sub(r"<[^>]+>", "", texto)  # Elimina etiquetas HTML
    texto = replace_emoji(texto, replace="")  # Elimina emojis
    texto = re.sub(r"[^a-záéíóúñü0-9\s]", "", texto)  # Elimina signos y símbolos raros
    texto = re.sub(r"\s+", " ", texto)  # Sustituye múltiples espacios por uno
    texto = texto.strip()  # Quita espacios al inicio/fin
    return texto


# ---------------------------
# Función: Aplicar limpieza
# ---------------------------
def procesar_archivo(ruta_csv):
    try:
        df = pd.read_csv(ruta_csv)
        validar_dataframe(df)

        print("[INFO] Eliminando duplicados exactos...")
        df = df.drop_duplicates(subset="texto")

        print("[INFO] Limpiando texto...")
        df["texto_limpio"] = df["texto"].apply(limpiar_texto)

        # Validación: No hay texto vacío luego de limpiar
        assert df["texto_limpio"].str.strip().astype(bool).all(), (
            "Texto vacío después de limpiar"
        )

        # Reporte comparativo
        original_chars = df["texto"].str.len().sum()
        limpio_chars = df["texto_limpio"].str.len().sum()
        print(f"[INFO] Caracteres totales originales: {original_chars}")
        print(f"[INFO] Caracteres después de limpiar: {limpio_chars}")
        print(
            f"[INFO] Diferencia: {original_chars - limpio_chars} caracteres eliminados"
        )

        # Guardar limpio
        df[["texto_limpio"]].to_csv("salida_limpia.csv", index=False)
        print("[✅] Archivo guardado como 'salida_limpia.csv'")

    except AssertionError as e:
        print(f"[❌ ERROR de validación] {e}")
    except Exception as e:
        print(f"[❌ ERROR inesperado] {e}")


# ---------------------------
# Punto de entrada del script
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python limpieza_texto.py entrada.csv")
        sys.exit(1)
    procesar_archivo(sys.argv[1])
