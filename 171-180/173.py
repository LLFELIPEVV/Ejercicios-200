# 🧠 Ejercicio 173/200 — Sanitización y validación segura para inferencia de texto
import re
import string


# Paso 1: Leer entrada desde consola
def obtener_entrada():
    entrada = input("🔎 Ingresa una frase para evaluar si es fake news:\n> ")
    return entrada.strip()


# Paso 2: Limpiar y sanitizar entrada
def sanitizar_texto(texto):
    # Eliminar etiquetas HTML/script y entidades
    texto = re.sub(r"<.*?>", "", texto)
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar signos de puntuación
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    # Remover números
    texto = re.sub(r"\d+", "", texto)
    # Quitar múltiples espacios
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# Paso 3: Validar entrada
def es_valido(texto):
    if len(texto) == 0:
        return False, "⚠️ Entrada vacía."

    palabras = texto.split()

    if len(palabras) < 3:
        return False, "⚠️ Muy corto. Necesitas al menos 3 palabras significativas."

    # Detectar repeticiones (más del 50% repetidas)
    repeticiones = sum(1 for w in set(palabras) if palabras.count(w) > 2)
    if repeticiones > 0:
        return False, "⚠️ Demasiadas repeticiones sospechosas."

    # Caracteres sin sentido
    if all(c in string.punctuation for c in texto):
        return False, "⚠️ Solo contiene símbolos."

    return True, ""


# Paso 4: Simular inferencia (modelo ficticio)
def simular_prediccion(texto):
    # Para el ejemplo: si contiene ciertas palabras clave, asumimos "fake"
    palabras_clave = [
        "increíble",
        "gratis",
        "vacunas causan",
        "conspiración",
        "no quieren que sepas",
    ]
    es_fake = any(clave in texto for clave in palabras_clave)
    return "🔴 Fake News" if es_fake else "🟢 Parece verídica"


# --------- MAIN ----------
if __name__ == "__main__":
    texto_raw = obtener_entrada()
    texto_limpio = sanitizar_texto(texto_raw)
    valido, error = es_valido(texto_limpio)

    if not valido:
        print(error)
    else:
        resultado = simular_prediccion(texto_limpio)
        print(f"\n✅ Texto limpio: {texto_limpio}")
        print(f"📊 Resultado de inferencia: {resultado}")
