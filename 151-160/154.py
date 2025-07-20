# 🧠 Ejercicio 154/200: Limpieza automática de texto desde archivo .txt y validación con assert
import re

# Ruta al archivo original
archivo_entrada = "limpieza_texto/noticias_crudas.txt"
archivo_salida = "limpieza_texto/noticias_limpias.txt"

# 1. Leer el archivo de entrada
with open(archivo_entrada, "r", encoding="utf-8") as f:
    lineas = f.readlines()


# 2. Limpiar cada línea
def limpiar_linea(texto):
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)  # Quitar símbolos no alfabéticos
    texto = re.sub(r"\s+", " ", texto).strip()  # Quitar espacios extra
    return texto


lineas_limpias = [limpiar_linea(linea) for linea in lineas]
lineas_limpias = [linea for linea in lineas_limpias if linea != ""]  # Eliminar vacías
lineas_limpias = list(set(lineas_limpias))  # Eliminar duplicados

# 3. Validaciones con assert
assert all(linea.strip() != "" for linea in lineas_limpias), "Hay líneas vacías"
assert len(lineas_limpias) == len(set(lineas_limpias)), "Hay duplicados"
assert all(len(linea.split()) >= 5 for linea in lineas_limpias), (
    "Hay líneas con menos de 5 palabras"
)

# 4. Guardar resultado
with open(archivo_salida, "w", encoding="utf-8") as f:
    for linea in lineas_limpias:
        f.write(linea + "\n")

print("✅ Archivo limpio guardado en:", archivo_salida)
