# ✅ Ejercicio 160/200 – Organización profesional: Script modular para evaluación de modelo con ROC, PR y matriz de confusión etiquetada
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)

# -------------------------------
# 🔧 Validaciones
# -------------------------------


def validate_inputs(y_true, y_probs):
    """
    Validación básica con assert
    """
    assert len(y_true) == len(y_probs), "Las listas deben tener la misma longitud"
    assert all(i in [0, 1] for i in y_true), "y_true solo debe contener 0 o 1"
    assert all(0.0 <= p <= 1.0 for p in y_probs), (
        "Las probabilidades deben estar entre 0 y 1"
    )


# -------------------------------
# 📈 ROC Curve
# -------------------------------


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR (Tasa de Falsos Positivos)")
    plt.ylabel("TPR (Tasa de Verdaderos Positivos)")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------
# 📉 Precision-Recall Curve
# -------------------------------


def plot_pr_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="darkorange")
    plt.xlabel("Recall (Sensibilidad)")
    plt.ylabel("Precisión")
    plt.title("Curva Precision-Recall")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------
# 🧩 Matriz de Confusión con etiquetas
# -------------------------------


def plot_confusion_matrix(y_true, y_pred, labels=["Real", "Fake"]):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Etiqueta predicha")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.show()


# -------------------------------
# ▶️ Ejecución del script (solo si se llama directamente)
# -------------------------------

if __name__ == "__main__":
    # Simulación de datos: en un proyecto real, se importarían desde archivos
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    y_probs = [0.1, 0.9, 0.8, 0.2, 0.95, 0.3, 0.85, 0.75, 0.1, 0.05]

    validate_inputs(y_true, y_probs)

    # Predicciones binarias por umbral 0.5
    y_pred = [1 if p >= 0.5 else 0 for p in y_probs]

    # Visualizaciones
    plot_roc_curve(y_true, y_probs)
    plot_pr_curve(y_true, y_probs)
    plot_confusion_matrix(y_true, y_pred)
