# 🧠 Ejercicio 148/200: Predicción desde archivo .txt con modelo .h5 en sistema de detección de fake news
import os
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.layers import TextVectorization

# 1. Ruta del archivo y modelo
RUTA_TXT = "noticia_prueba.txt"
RUTA_MODELO = "model_1.h5"

# 2. Validar existencia de archivos
assert os.path.exists(RUTA_TXT), f"❌ Archivo de texto no encontrado: {RUTA_TXT}"
assert os.path.exists(RUTA_MODELO), f"❌ Modelo no encontrado: {RUTA_MODELO}"

# 3. Leer el texto desde archivo .txt
with open(RUTA_TXT, "r", encoding="utf-8") as f:
    texto = f.read().strip()

assert len(texto) > 10, "❌ El texto es demasiado corto para predecir"

# 4. Cargar corpus base para adaptar vectorizador
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")

corpus = pd.concat([df_fake, df_true])["text"].dropna().astype(str).tolist()

# 5. Crear y adaptar vectorizador
vectorizador = TextVectorization(max_tokens=1000, output_mode="tf_idf")
vectorizador.adapt(corpus)

# 6. Vectorizar el texto de entrada
vector = vectorizador(np.array([texto]))  # Se espera una lista

# 7. Cargar modelo entrenado
modelo = load_model(RUTA_MODELO)

# 8. Realizar predicción
prob = modelo.predict(vector, verbose=0).flatten()[0]
pred_binaria = int(np.round(prob))

# 9. Mostrar resultado profesional
print("=== Resultado de la predicción ===")
print(f"📝 Texto: {texto[:75]}...")
print(f"📊 Probabilidad de ser REAL: {prob:.4f}")
print("✅ Clasificación:", "REAL ✅" if pred_binaria == 1 else "FAKE ❌")

# 10. Validaciones con assert
assert 0.0 <= prob <= 1.0, "❌ La probabilidad está fuera del rango válido"
assert pred_binaria in [0, 1], "❌ La predicción no es binaria"
print("🔒 Validación superada: predicción válida y binaria.")
