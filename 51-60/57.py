# 🧠 Ejercicio 57/200: Detección de titulares clickbait con GloVe + Similitud Coseno
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Optional, List


# 🧬 Cargar los vectores GloVe desde archivo
def cargar_embeddings(path: str) -> Dict[str, np.ndarray]:
    embeddings = {}
    with open(path, encoding="utf8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embeddings[palabra] = vector
    return embeddings


# 🧠 Convierte un texto a un vector promedio de palabras con embeddings
def texto_a_vector(
    texto: str, embeddings: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    palabras = texto.lower().split()
    vectores = [embeddings[p] for p in palabras if p in embeddings]
    if not vectores:
        return None
    return np.mean(vectores, axis=0)


# 🧪 Compara un titular y su cuerpo de noticia con similitud coseno
def evaluar_incoherencia(
    noticia: Dict[str, str], embeddings: Dict[str, np.ndarray]
) -> Optional[float]:
    v_headline = texto_a_vector(noticia["headline"], embeddings)
    v_body = texto_a_vector(noticia["body"], embeddings)

    if v_headline is None or v_body is None:
        return None  # No hay suficientes palabras con embedding
    return cosine_similarity([v_headline], [v_body])[0][0]


# 📊 Mostrar resultados con mensaje de interpretación
def mostrar_resultados(
    noticias: List[Dict[str, str]],
    embeddings: Dict[str, np.ndarray],
    umbral: float = 0.6,
):
    for i, noticia in enumerate(noticias, 1):
        similitud = evaluar_incoherencia(noticia, embeddings)

        print(f"\n📰 Noticia {i}:")
        print(f"🔖 Titular: {noticia['headline']}")
        print(f"📄 Cuerpo: {noticia['body']}")
        if similitud is None:
            print("❌ Insuficiente información semántica para analizar.")
        else:
            print(f"📉 Similitud coseno: {similitud:.4f}")
            if similitud < umbral:
                print("⚠️ Resultado: Posible *clickbait* o titular incongruente ⚠️")
            else:
                print("✅ Resultado: Titular coherente con el cuerpo de la noticia")


# 📚 Noticias de ejemplo (con intencionalidad de clickbait)
noticias = [
    {
        "headline": "¿Sabías que Naruto es en realidad un alienígena? ¡La verdad te dejará sin palabras!",
        "body": "Una teoría de fans sugiere que Naruto Uzumaki podría tener orígenes extraterrestres debido a su chakra, pero no hay confirmación oficial.",
    },
    {
        "headline": "¡El episodio final de Dragon Ball Super revela el secreto más grande de Goku! ¡No podrás creerlo!",
        "body": "El final mostró a Goku dominando el Ultra Instinto, pero no se revelaron secretos nuevos de su pasado.",
    },
    {
        "headline": "¡Los fans de One Piece descifran el significado de 'D.' y es más simple de lo que crees!",
        "body": "Teorías de fans sugieren significados, pero Eiichiro Oda aún no ha revelado nada oficialmente.",
    },
    {
        "headline": "Vacuna contra el COVID-19 demuestra eficacia del 95% en pruebas clínicas.",
        "body": "Pfizer y BioNTech anunciaron resultados prometedores en su vacuna durante la fase 3 del ensayo clínico.",
    },
    {
        "headline": "Científicos descubren nuevo planeta potencialmente habitable en sistema cercano.",
        "body": "Astrónomos detectaron un exoplaneta en zona habitable de Proxima Centauri, el sistema más cercano a la Tierra.",
    },
    {
        "headline": "¡El crossover de anime definitivo se acerca y cambiará todo para siempre!",
        "body": "Rumores surgieron por colaboraciones en videojuegos, pero no hay anuncios oficiales de un crossover real.",
    },
    {
        "headline": "¡Descubre el poder oculto de los Saiyajin que ni siquiera Goku conoce! ¡Te dejará boquiabierto!",
        "body": "Un blog no oficial menciona habilidades teóricas de los Saiyajin, pero nada forma parte del canon.",
    },
]


# 🧠 Ejecutar detección con umbral definido
embedding_index = cargar_embeddings("Gloove/glove.6B.100d.txt")
mostrar_resultados(noticias, embedding_index, umbral=0.6)
