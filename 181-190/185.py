# 🛡️ Ejercicio 185/200 — Detección de Entradas Maliciosas o Inusuales en Inferencia de Texto
import re
import html

from collections import Counter


# 1. Limpieza y tokenización simple
def sanitizar(texto):
    # Decodifica HTML y remueve etiquetas
    texto = html.unescape(texto)
    texto = re.sub(r"<[^>]+>", "", texto)

    # Normaliza y deja solo letras
    texto = re.sub(r"[^a-zA-Z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip().lower()
    return texto


# 2. Reglas de validación
def validar(texto):
    tokens = texto.split()
    n = len(tokens)

    # Regla 1: Texto demasiado corto
    if n < 3:
        return False, "Texto demasiado corto"

    # Regla 2: Repetición excesiva
    frecuencia = Counter(tokens)
    max_repe = max(frecuencia.values())
    if max_repe / n > 0.6:
        return False, "Repetición excesiva de palabras"

    # Regla 3: Caracteres extraños en el original
    if re.search(r"[^\w\s.,!?¿¡]", texto_original):
        return False, "Caracteres sospechosos detectados"

    # Regla 4: Letras repetidas como spam
    if re.search(r"(.)\1{3,}", texto_original):
        return False, "Patrón adversario repetido (letras repetidas)"

    return True, "Apta para inferencia"


# 3. Simulación de entradas
entradas = [
    "Click here to win $$$$ FREE IPHONE $$$",
    "asdfasdfasdfasdfasdf",
    "The president held a press conference today",
    "government government government government government",
    "",
    "AI is changing the world",
]

# 4. Ejecución
for i, entrada in enumerate(entradas):
    print(f"\n📥 Entrada {i + 1}: {entrada}")

    texto_original = entrada  # para revisar símbolos extraños
    texto_limpio = sanitizar(entrada)

    ok, motivo = validar(texto_limpio)
    estado = "✅ Apta para inferencia" if ok else "❌ Rechazada"
    print(f"Resultado: {estado} — {motivo}")
