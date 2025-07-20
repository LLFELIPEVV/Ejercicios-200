# 🧠 Ejercicio 172/200 — Usar embeddings preentrenados (GloVe) sin reentrenar
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# Paso 1: Leer frases desde archivo CSV
def cargar_frases_csv(ruta):
    frases = []
    with open(ruta, encoding="utf-8") as f:
        reader = csv.reader(f)
        for fila in reader:
            if fila:  # si la fila no está vacía
                frases.append(fila[0].strip().lower())
    return list(set(frases))  # eliminar duplicados


# Paso 2: Limpiar texto de símbolos y caracteres no alfabéticos
def limpiar_texto(textos):
    return [re.sub(r"[^a-záéíóúñü\s]", "", t.lower()) for t in textos]


# Paso 3: Cargar embeddings de GloVe
def cargar_glove(ruta_glove):
    embeddings = {}
    with open(ruta_glove, encoding="utf-8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embeddings[palabra] = vector
    return embeddings


# Paso 4: Obtener embedding promedio de cada frase
def vectorizar_frases(frases, embeddings, dimension=50):
    vectores = []
    frases_filtradas = []
    for frase in frases:
        palabras = frase.split()
        vectores_palabras = [embeddings[p] for p in palabras if p in embeddings]
        if vectores_palabras:
            promedio = np.mean(vectores_palabras, axis=0)
            vectores.append(promedio)
            frases_filtradas.append(frase)
    return frases_filtradas, np.array(vectores)


# Paso 5: Aplicar PCA
def reducir_con_pca(vectores, componentes=2):
    pca = PCA(n_components=componentes, random_state=42)
    return pca.fit_transform(vectores)


# Paso 6: Visualizar en scatter plot
def graficar(frases, vectores_2d):
    plt.figure(figsize=(10, 8))
    for i, frase in enumerate(frases):
        x, y = vectores_2d[i]
        plt.scatter(x, y)
        plt.text(
            x + 0.01,
            y + 0.01,
            frase[:40] + ("..." if len(frase) > 40 else ""),
            fontsize=9,
        )
    plt.title("Frases representadas con GloVe + PCA")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------- MAIN ----------
if __name__ == "__main__":
    try:
        frases = cargar_frases_csv("frases.csv")
        frases = limpiar_texto(frases)
        embeddings = cargar_glove("glove.6B.50d.txt")  # archivo necesario
        frases_final, vectores = vectorizar_frases(frases, embeddings)

        assert vectores.shape[0] == len(frases_final), (
            "No se vectorizaron todas las frases"
        )
        vectores_2d = reducir_con_pca(vectores)

        graficar(frases_final, vectores_2d)

    except FileNotFoundError as e:
        print(f"⚠️ Archivo no encontrado: {e}")
    except Exception as err:
        print(f"❌ Error inesperado: {err}")
