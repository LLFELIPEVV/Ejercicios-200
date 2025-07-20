# 🧪 Ejercicio 192/200 — Detección de entradas adversarias simples para proteger tu modelo
import re

# 1. Simulación de un vocabulario obtenido de entrenamiento
vocabulario_entrenado = {
    "government",
    "fake",
    "news",
    "real",
    "nasa",
    "economy",
    "cure",
    "doctor",
    "space",
    "mission",
    "believe",
    "scientists",
}

# 2. Lista de palabras comunes en inglés para evitar falsos positivos
stopwords = {"the", "is", "and", "on", "this", "you", "by", "a", "of", "to"}


# 3. Función principal de validación
def validar_entrada(texto):
    print(f"\n📩 Verificando entrada: {texto}")
    texto = texto.strip().lower()

    # (1) Longitud mínima y máxima permitida
    if len(texto) < 10 or len(texto) > 300:
        print("❌ Entrada rechazada: Longitud no válida")
        return False

    # (2) Solo mayúsculas o solo símbolos
    if texto.isupper() or re.fullmatch(r"[^\w\s]+", texto):
        print("❌ Entrada rechazada: Solo mayúsculas o símbolos")
        return False

    # (3) Tokenización simple
    palabras = re.findall(r"\b\w+\b", texto)

    # (4) Rechazar si más del 50% de palabras no están en el vocabulario (ignorando stopwords)
    palabras_validas = [
        p for p in palabras if p in vocabulario_entrenado or p in stopwords
    ]
    if len(palabras_validas) / max(len(palabras), 1) < 0.5:
        print("❌ Entrada rechazada: Demasiadas palabras desconocidas")
        return False

    # (5) Detectar letras repetidas artificialmente (e.g., "faaaaake")
    for palabra in palabras:
        if re.search(r"(.)\1{3,}", palabra):  # Letra repetida más de 3 veces
            print(
                f"❌ Entrada rechazada: Palabra con repetición sospechosa → {palabra}"
            )
            return False

    print("✅ Entrada aceptada")
    return True


# 4. Casos de prueba
entradas = [
    "BREAKING NEWS: THE GOVERNMENT IS LYING",  # Solo mayúsculas
    "govrnmt spspp lofff fgfffffff",  # Letras inventadas
    "Faaaaake news spreading now!!!",  # Repetición
    "Real scientists confirm space discovery",  # Correcta
    "!!!###$$$???",  # Solo símbolos
    "believe you by cure by you",  # Ambigua, pero válida
]

# 5. Ejecutar validaciones
for entrada in entradas:
    validar_entrada(entrada)
