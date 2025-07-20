# 🧠 Ejercicio 169/200 — Ensemble Liviano por Votación Soft desde CSV
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)


# -----------------------
# Limpieza básica de texto
# -----------------------
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñ0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


# -----------------------
# Validación de archivo
# -----------------------
def validar_entrada(df):
    assert "texto" in df.columns and "etiqueta" in df.columns, (
        "Se requieren columnas: texto, etiqueta"
    )
    assert df["etiqueta"].isin([0, 1]).all(), (
        "Las etiquetas deben ser 0 (real) o 1 (fake)"
    )
    print(f"[INFO] Archivo válido. Total registros: {len(df)}")


# -----------------------
# Gráficos de evaluación
# -----------------------
def graficar_metricas(y_true, y_prob, y_pred):
    os.makedirs("metricas", exist_ok=True)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Matriz de Confusión")
    plt.savefig("metricas/matriz_confusion.png")
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid()
    plt.savefig("metricas/curva_roc.png")
    plt.close()

    # Curva PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precisión")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.grid()
    plt.savefig("metricas/curva_pr.png")
    plt.close()

    print("[✔] Métricas visuales guardadas en /metricas")


# -----------------------
# Ejecución principal
# -----------------------
def main(ruta_csv):
    try:
        df = pd.read_csv(ruta_csv)
        validar_entrada(df)

        # Limpieza
        df.drop_duplicates(subset="texto", inplace=True)
        df["texto"] = df["texto"].astype(str).apply(limpiar_texto)

        X = df["texto"]
        y = df["etiqueta"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Modelos base
        modelo_lr = LogisticRegression(max_iter=1000)
        modelo_nb = MultinomialNB()
        modelo_svc = CalibratedClassifierCV(LinearSVC())

        # Voting Soft
        ensemble = VotingClassifier(
            estimators=[("lr", modelo_lr), ("nb", modelo_nb), ("svc", modelo_svc)],
            voting="soft",
        )

        # Pipeline completo
        pipeline = Pipeline(
            [("vectorizador", TfidfVectorizer()), ("clasificador", ensemble)]
        )

        print("[⚙️] Entrenando ensemble soft voting...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probabilidad de clase 1 (fake)

        graficar_metricas(y_test, y_prob, y_pred)

    except AssertionError as e:
        print(f"[❌ Error de validación] {e}")
    except Exception as e:
        print(f"[❌ Error inesperado] {e}")


# -----------------------
# Entrada por consola
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python ensemble_soft_voting.py archivo.csv")
        sys.exit(1)
    main(sys.argv[1])
