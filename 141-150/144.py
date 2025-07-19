# 🧠 Ejercicio 144/200: Parsing y predicción desde archivos .json para entrada realista de usuario
# -*- coding: utf-8 -*-
import pandas as pd

from keras.models import load_model

# Cargar el modelo previamente entrenado
model = load_model("modelo_fake_news.h5")

# Leer archivo JSON con los textos
# Cargar CSV
df_fake = pd.read_csv(r"Datasets\archive\Fake.csv")
df_true = pd.read_csv(r"Datasets\archive\True.csv")

# Etiquetar
df_fake["label"] = 0
df_true["label"] = 1

# Unir y mezclar
data = (
    pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
)

# Asegurarse de que la clave 'texts' exista
assert "text" in data, "El archivo JSON debe contener la clave 'texts'"

# Lista de textos de entrada
input_texts = data["text"]

# Verifica que sean cadenas de texto
assert all(isinstance(t, str) for t in input_texts), (
    "Todos los textos deben ser strings"
)

# Realizar las predicciones
predicciones = model.predict(input_texts)

# Interpretar y mostrar resultados
for i, texto in enumerate(input_texts):
    probabilidad = predicciones[i][0]
    etiqueta = "FAKE" if probabilidad >= 0.5 else "REAL"
    print(f"\nTexto {i + 1}:")
    print(f"Contenido: {texto}")
    print(f"Predicción: {etiqueta} ({probabilidad:.2f})")
