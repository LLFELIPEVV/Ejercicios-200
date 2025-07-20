# 🧠 Ejercicio 174/200 — Detección de entradas adversarias simples (textos camuflados)
# coding: utf-8
import re
import unicodedata

# Mapa básico de Leetspeak para reemplazo
LEET_MAP = {"4": "a", "3": "e", "1": "i", "0": "o", "7": "t", "5": "s"}


# Paso 1: Convertir leetspeak a texto plano
def convertir_leetspeak(texto):
    return "".join(LEET_MAP.get(c, c) for c in texto)


# Paso 2: Normalizar unicode a caracteres planos (e.g. eliminar tildes y similares)
def normalizar_unicode(texto):
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


# Paso 3: Eliminar repeticiones exageradas
def reducir_repeticiones(texto):
    return re.sub(r"(.)\1{2,}", r"\1\1", texto)


# Paso 4: Limpiar texto y detectar manipulación
def limpiar_y_detectar(texto_original):
    texto = texto_original.lower()
    texto = normalizar_unicode(texto)
    texto = convertir_leetspeak(texto)
    texto = reducir_repeticiones(texto)
    texto = re.sub(r"[^a-z\s]", "", texto)  # eliminar todo excepto letras y espacios

    # Comparar cuántos cambios se hicieron
    cambios = sum(1 for a, b in zip(texto_original, texto) if a != b)
    if cambios > 5:
        return None, True  # Se considera entrada adversaria

    return texto, False


# Simulación de inferencia segura
def simular_inferencia(texto):
    palabras_clave = [
        "vacunas causan",
        "conspiración",
        "gobierno",
        "no quieren que sepas",
    ]
    es_fake = any(clave in texto for clave in palabras_clave)
    return "🔴 Fake News" if es_fake else "🟢 Parece verídica"


# -------- MAIN --------
if __name__ == "__main__":
    entrada_raw = input("🔎 Ingresa una frase para analizar:\n> ").strip()
    texto_limpio, es_adversario = limpiar_y_detectar(entrada_raw)

    if es_adversario:
        print("⚠️ Entrada sospechosa: contiene manipulación textual adversaria.")
        print("🔒 Por seguridad, la inferencia fue cancelada.")
    else:
        print(f"\n✅ Texto procesado: {texto_limpio}")
        resultado = simular_inferencia(texto_limpio)
        print(f"📊 Resultado: {resultado}")
