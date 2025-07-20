# 🚀 Ejercicio 186/200 — Pipeline de Inferencia Segura con Embeddings Keras
import re
import html

from collections import Counter
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, TextVectorization


# 1. Sanitización del texto
def sanitizar(texto):
    texto = html.unescape(texto)
    texto = re.sub(r"<[^>]+>", "", texto)
    texto = re.sub(r"[^a-zA-Z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip().lower()
    return texto


# 2. Validación contra patrones adversarios
def validar(texto_limpio, texto_original):
    tokens = texto_limpio.split()
    if len(tokens) < 3:
        return False, "Texto demasiado corto"
    frecuencia = Counter(tokens)
    if max(frecuencia.values()) / len(tokens) > 0.6:
        return False, "Repetición excesiva"
    if re.search(r"[^\w\s.,!?¿¡]", texto_original):
        return False, "Caracteres sospechosos"
    if re.search(r"(.)\1{3,}", texto_original):
        return False, "Patrón adversario"
    return True, "Entrada válida"


# 3. Simula entrada del usuario
entrada_usuario = "Breaking: the president held a secret meeting today"

# 4. Limpieza y validación
texto_original = entrada_usuario
texto_limpio = sanitizar(texto_original)
es_valido, razon = validar(texto_limpio, texto_original)

if not es_valido:
    print(f"❌ Entrada rechazada: {razon}")
    exit()

print("✅ Entrada aceptada → Iniciando pipeline...")

# 5. Tokenización
tokenizer = TextVectorization(max_tokens=1000)
tokenizer.adapt([texto_limpio])
secuencia = tokenizer.texts_to_sequences([texto_limpio])
secuencia_padded = pad_sequences(secuencia, maxlen=10, padding="post")

# 6. Modelo dummy para simular inferencia
modelo = Sequential(
    [
        Embedding(input_dim=1000, output_dim=8, input_length=10),
        GlobalAveragePooling1D(),
        Dense(1, activation="sigmoid"),  # salida binaria: real/fake
    ]
)
modelo.compile(optimizer="adam", loss="binary_crossentropy")

# 7. Simula pesos aleatorios (no entrenado)
pred = modelo.predict(secuencia_padded, verbose=0)[0][0]

# 8. Resultado final
clase = "🟢 Real" if pred >= 0.5 else "🔴 Fake"
print(f"📊 Predicción simulada: {clase} ({pred:.3f})")
