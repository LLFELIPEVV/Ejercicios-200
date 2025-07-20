# 🧠 Ejercicio 168/200 — Visualización Profesional de Métricas de Clasificación
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)


# -----------------------------
# Validación inicial del DataFrame
# -----------------------------
def validar_csv(df):
    columnas = {"y_true", "y_pred", "y_prob"}
    assert columnas.issubset(df.columns), (
        "El archivo debe contener las columnas: y_true, y_pred, y_prob"
    )
    assert df["y_true"].isin([0, 1]).all(), "y_true debe contener solo 0 o 1"
    assert df["y_pred"].isin([0, 1]).all(), "y_pred debe contener solo 0 o 1"
    assert df["y_prob"].between(0, 1).all(), "y_prob debe estar entre 0 y 1"
    print(f"[INFO] Datos válidos. Registros: {len(df)}")


# -----------------------------
# Matriz de confusión
# -----------------------------
def graficar_matriz_confusion(y_true, y_pred, etiquetas, ruta_salida):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Matriz de Confusión")
    plt.savefig(ruta_salida)
    plt.close()
    print(f"[✔] Matriz de confusión guardada en: {ruta_salida}")


# -----------------------------
# Curva ROC
# -----------------------------
def graficar_roc(y_true, y_prob, ruta_salida):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Azar")
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid()
    plt.savefig(ruta_salida)
    plt.close()
    print(f"[✔] Curva ROC guardada en: {ruta_salida}")


# -----------------------------
# Curva Precision-Recall
# -----------------------------
def graficar_precision_recall(y_true, y_prob, ruta_salida):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    plt.xlabel("Recall (Sensibilidad)")
    plt.ylabel("Precisión")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.grid()
    plt.savefig(ruta_salida)
    plt.close()
    print(f"[✔] Curva PR guardada en: {ruta_salida}")


# -----------------------------
# Ejecución principal
# -----------------------------
def ejecutar_metricas(ruta_csv):
    try:
        df = pd.read_csv(ruta_csv)
        validar_csv(df)

        y_true = df["y_true"]
        y_pred = df["y_pred"]
        y_prob = df["y_prob"]

        os.makedirs("metricas", exist_ok=True)
        graficar_matriz_confusion(
            y_true, y_pred, ["REAL", "FAKE"], "metricas/matriz_confusion.png"
        )
        graficar_roc(y_true, y_prob, "metricas/curva_roc.png")
        graficar_precision_recall(y_true, y_prob, "metricas/curva_pr.png")

    except AssertionError as e:
        print(f"[❌ Error de validación] {e}")
    except Exception as e:
        print(f"[❌ Error inesperado] {e}")


# -----------------------------
# Punto de entrada
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python graficar_metricas.py archivo_resultados.csv")
        sys.exit(1)
    ejecutar_metricas(sys.argv[1])
