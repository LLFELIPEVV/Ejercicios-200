# 🧠 Ejercicio 155/200 — Ensemble por Votación con Validación y Visualización Profesional

# 🗂️ Estructura sugerida del archivo
# 📁 proyecto_ensamble/
# │
# ├── main.py               # Punto de entrada
# ├── preprocess.py         # Función de carga y limpieza
# ├── models.py             # Definición del ensemble
# ├── evaluation.py         # Visualización y métricas
# └── fake_news_sample.csv  # Dataset pequeño de ejemplo


from preprocess import cargar_datos
from models import construir_ensemble
from evaluation import mostrar_matriz_confusion, graficar_curva_roc

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# 1. Cargar datos
X_raw, y = cargar_datos("fake_news_sample.csv")

# 2. Vectorizar texto
vectorizador = TfidfVectorizer(max_features=1000)
X = vectorizador.fit_transform(X_raw)

# 3. Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Construir ensemble
modelo = construir_ensemble()
modelo.fit(X_train, y_train)

# 5. Predicciones
y_pred = modelo.predict(X_test)

# 6. Evaluación textual
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Visualizaciones
mostrar_matriz_confusion(y_test, y_pred)
graficar_curva_roc(modelo.estimators_[1], X_test, y_test)  # LogisticRegression
