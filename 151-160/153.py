# ✅ Ejercicio 153/200 — Validación y manejo de errores para predicciones con entrada de consola
import re


def clean_text(text):
    """
    Limpia el texto eliminando símbolos, convirtiendo a minúsculas y quitando espacios extra.
    """
    text = text.lower()  # Minúsculas
    text = re.sub(
        r"[^a-zA-Z\s]", "", text
    )  # Eliminar todo lo que no sea letras ni espacios
    text = re.sub(r"\s+", " ", text).strip()  # Quitar espacios múltiples
    return text


def predict_fake_news(text):
    """
    Simula una predicción simple basada en palabras clave.
    No requiere modelo entrenado.
    """
    fake_keywords = ["conspiración", "vacuna", "gobierno", "secreto"]
    for word in fake_keywords:
        if word in text:
            return "⚠️ Posible noticia FALSA"
    return "✅ Parece una noticia legítima"


def main():
    try:
        raw_input = input("🔎 Ingresa el texto de la noticia:\n> ")

        # Validación
        assert isinstance(raw_input, str), "La entrada debe ser texto."
        assert raw_input.strip() != "", "No puedes ingresar un texto vacío."

        # Limpieza del texto
        cleaned = clean_text(raw_input)
        print(f"\n🧹 Texto limpio: {cleaned}\n")

        # Simulación de predicción
        result = predict_fake_news(cleaned)
        print(f"📊 Resultado del sistema: {result}")

    except AssertionError as ae:
        print(f"❌ Error de validación: {ae}")

    except Exception as e:
        print(f"❌ Error inesperado: {e}")


if __name__ == "__main__":
    main()
