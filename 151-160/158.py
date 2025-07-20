# 🧠 Ejercicio 158/200: Ensemble ligero con votación y visualización de métricas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc

# === 1. Cargar dataset desde CSV ===
try:
    df = pd.read_csv("fake_news_sample.csv")
except FileNotFoundError:
    print("⚠️ Archivo 'fake_news_sample.csv' no encontrado.")
    exit()

# === 2. Limpieza básica del texto ===
df.drop_duplicates(subset="text", inplace=True)
df["text"] = df["text"].astype(str).str.replace(r"\W+", " ", regex=True).str.lower()
df["label"] = df["label"].astype(int)

# === 3. División de datos ===
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, stratify=df["label"], random_state=42
)

# === 4. Vectorización ===
vectorizer = CountVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Verificamos dimensiones
assert X_train_vec.shape[0] == len(y_train), "Tamaño inconsistente en entrenamiento"
assert X_test_vec.shape[0] == len(y_test), "Tamaño inconsistente en prueba"

# === 5. Crear clasificadores simples ===
clf1 = LogisticRegression(max_iter=1000)
clf2 = MultinomialNB()
clf3 = DecisionTreeClassifier(max_depth=5)

# === 6. Ensemble por votación blanda ===
ensemble = VotingClassifier(
    estimators=[("lr", clf1), ("nb", clf2), ("dt", clf3)], voting="soft"
)
ensemble.fit(X_train_vec, y_train)

# === 7. Predicciones ===
y_pred = ensemble.predict(X_test_vec)
y_prob = ensemble.predict_proba(X_test_vec)[:, 1]  # Probabilidades clase 1

# === 8. Métricas ===
# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("conf_matrix.png")
plt.close()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

print("✅ Ensemble entrenado y evaluado. Resultados guardados como imágenes.")
