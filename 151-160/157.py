# 🧠 Ejercicio 157/200 — Visualización Profesional de Curvas ROC y PR con Etiquetas Personalizadas
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
)

# ===================== 1. Cargar el CSV =====================
file_path = "fake_news_sample.csv"

# Si no existe el archivo, creamos un dataset artificial
if not os.path.exists(file_path):
    df = pd.DataFrame(
        {
            "text": [
                "Breaking news! COVID is fake.",
                "Scientists confirm moon has water.",
                "Aliens land in New York.",
                "Vaccine saves millions.",
                "New planet discovered in solar system.",
            ]
            * 40,  # 200 ejemplos
            "label": [1, 0, 1, 0, 0] * 40,
        }
    )
    df.to_csv(file_path, index=False)
else:
    df = pd.read_csv(file_path)

assert df.shape[0] >= 100, "El dataset debe tener al menos 100 ejemplos."
assert df["label"].nunique() == 2, "El dataset debe tener exactamente 2 clases."


# ===================== 2. Limpieza del texto =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["text"] = df["text"].apply(clean_text)

# ===================== 3. Vectorización =====================
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# ===================== 4. División y Entrenamiento =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = MultinomialNB()
model.fit(X_train, y_train)

# ===================== 5. Predicciones =====================
y_probs = model.predict_proba(X_test)[:, 1]  # Solo clase positiva

# ===================== 6. ROC =====================
fpr, tpr, thresholds_roc = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Modelo Naive Bayes")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# ===================== 7. Precision-Recall =====================
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label="Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall - Modelo Naive Bayes")
plt.grid(True)
plt.tight_layout()
plt.savefig("pr_curve.png")
plt.close()

print("✅ Curvas ROC y PR guardadas como imágenes.")
