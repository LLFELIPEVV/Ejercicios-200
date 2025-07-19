# ✅ Ejercicio 136/200 – Validación profesional de modelos con pruebas unitarias básicas en TensorFlow
import unittest
import numpy as np

from keras.models import load_model

# --- Carga del modelo guardado previamente ---
# Suponemos que ya entrenaste y guardaste un modelo llamado 'modelo_fake_news.h5'
modelo = load_model("modelo_fake_news.h5")

# --- Simulamos un lote de entrada válido ---
X_test = np.random.rand(
    10, 100
)  # 10 muestras, 100 características (ajusta según tu modelo)
y_test = np.random.randint(0, 2, size=(10, 1))  # Clases binarias: 0 o 1


# --- Creamos una clase de prueba unitaria ---
class TestModeloFakeNews(unittest.TestCase):
    def test_modelo_se_carga(self):
        """Verifica que el modelo se haya cargado correctamente."""
        self.assertIsNotNone(modelo)

    def test_forma_prediccion(self):
        """Verifica que las predicciones tengan la forma esperada."""
        predicciones = modelo.predict(X_test)
        self.assertEqual(
            predicciones.shape[0], X_test.shape[0]
        )  # Mismo número de muestras

    def test_valores_validos(self):
        """Verifica que el modelo produce predicciones dentro de rango [0,1]."""
        predicciones = modelo.predict(X_test)
        self.assertTrue(np.all(predicciones >= 0) and np.all(predicciones <= 1))

    def test_accuracy_basico(self):
        """Verifica que el modelo tenga una precisión mayor que 0 en un lote."""
        loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
        self.assertGreater(acc, 0.0)


# --- Punto de entrada del script ---
if __name__ == "__main__":
    unittest.main()
