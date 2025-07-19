# ✅ Ejercicio 140/200 – Validación profesional de un sistema con pruebas unitarias básicas para detección de noticias falsas
import unittest
import numpy as np

from keras.models import load_model

# Carga del modelo previamente entrenado y guardado
MODEL_PATH = "modelo_fake_news.h5"

# Simulamos 3 entradas válidas (ya vectorizadas) y 1 inválida
x_validas = np.array(
    [
        np.random.rand(100),  # Simulación de entrada 1
        np.random.rand(100),  # Entrada 2
        np.random.rand(100),  # Entrada 3
    ]
)
x_invalida = np.array(["texto sin vectorizar"])  # Entrada malformada


class TestFakeNewsModel(unittest.TestCase):
    def setUp(self):
        """Se ejecuta antes de cada test individual"""
        self.model = load_model(MODEL_PATH)

    def test_modelo_carga(self):
        """Verifica que el modelo se cargó correctamente"""
        self.assertIsNotNone(self.model)

    def test_prediccion_valida(self):
        """Verifica que la predicción no lanza errores"""
        try:
            preds = self.model.predict(x_validas)
            self.assertEqual(preds.shape[0], 3)
        except Exception as e:
            self.fail(f"Error en predicción válida: {e}")

    def test_salida_rango(self):
        """Verifica que las salidas estén en el rango [0,1]"""
        preds = self.model.predict(x_validas)
        for p in preds:
            self.assertTrue(0 <= p[0] <= 1, f"Predicción fuera de rango: {p[0]}")

    def test_entrada_invalida(self):
        """Verifica que el modelo lanza un error con entrada incorrecta"""
        with self.assertRaises(Exception):
            self.model.predict(x_invalida)


if __name__ == "__main__":
    # Ejecuta todos los tests
    unittest.main()
