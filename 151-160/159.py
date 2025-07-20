# ✅ Ejercicio 159/200 — Ensemble por votación liviano (VotingClassifier) con validación assert y lectura desde .csv
import re
import pandas as pd
import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Leer archivo .csv
try:
    df = pd.read_csv("noticias.csv")
except FileNotFoundError:
    print("⚠️ Error: El archivo 'noticias.csv' no fue encontrado.")
    exit()


# 2. Limpieza automática básica del texto
def limpiar_texto(texto):
    texto = texto.lower()  # minúsculas
    texto = re.sub(r"[^a-zA-Z0-9\s]", "", texto)  # eliminar símbolos
    texto = re.sub(r"\s+", " ", texto).strip()  # espacios repetidos
    return texto


df["texto"] = df["texto"].astype(str).apply(limpiar_texto)
df.drop_duplicates(subset="texto", inplace=True)  # quitar duplicados exactos

# 3. Vectorización de texto
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["texto"])
y = df["etiqueta"]

# 4. División de datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Definición de modelos livianos
modelo_nb = MultinomialNB()
modelo_lr = LogisticRegression(max_iter=1000)
modelo_dt = DecisionTreeClassifier(max_depth=3)  # árbol pequeño

# 6. Ensemble por votación (mayoría)
ensemble = VotingClassifier(
    estimators=[("nb", modelo_nb), ("lr", modelo_lr), ("dt", modelo_dt)], voting="hard"
)

# 7. Entrenar y predecir
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# 8. Validaciones básicas
assert len(y_pred) == len(y_test), (
    "❌ Error: la predicción no coincide con la cantidad de ejemplos"
)
assert set(np.unique(y_pred)).issubset({0, 1}), (
    "❌ Error: las predicciones contienen etiquetas inválidas"
)

# 9. Evaluación
acc = accuracy_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)

print("\n✅ Accuracy del ensemble:", round(acc, 3))
print("📊 Matriz de Confusión:")
print(matriz)

# 10. Etiquetas interpretables para humanos
print("\n🔎 Leyenda:")
print("Fila = verdadero | Columna = predicho")
print("Fila 0 = Noticias Falsas")
print("Fila 1 = Noticias Reales")
