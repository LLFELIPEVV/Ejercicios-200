# 🧠 Ejercicio 121/200: 🧹 Limpieza avanzada de texto con expresiones regulares (re)
# Paso 1: Importar librerías necesarias
import re
import pandas as pd

# Paso 2: Simular un pequeño dataset textual
data = {
    "titular": [
        "🔥¡Última hora! <b>Gobierno</b> anuncia nuevas medidas ➡️ https://t.co/abc123",
        "@usuario Mira esto: #FakeNews en aumento en todo el país!",
        "Visita nuestro sitio web 👉 www.fake-site.com 🛑 y entérate",
        "“Presidente”: ‘Los datos NO cuadran’!!! 😡😡 #crisis",
    ]
}

df = pd.DataFrame(data)


# Paso 3: Definir la función de limpieza con expresiones regulares
def limpiar_texto(texto):
    # Eliminar etiquetas HTML
    texto = re.sub(r"<.*?>", "", texto)

    # Eliminar URLs
    texto = re.sub(r"http\S+|www\.\S+", "", texto)

    # Eliminar menciones y hashtags
    texto = re.sub(r"@\w+|#\w+", "", texto)

    # Eliminar emojis y símbolos no alfabéticos (solo conserva letras, números y espacios)
    texto = re.sub(r"[^\w\sáéíóúÁÉÍÓÚüÜñÑ]", "", texto)

    # Reemplazar múltiples espacios por uno solo
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


# Paso 4: Aplicar la limpieza al DataFrame
df["limpio"] = df["titular"].apply(limpiar_texto)
# Paso 5: Visualizar los resultados
print(df)
