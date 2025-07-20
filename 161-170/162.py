# 🧠 Ejercicio 162/200 — Entrenamiento con Regresión Logística + Visualización de Métricas
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)


# -------------------------------
# Función: cargar y validar datos
# -------------------------------
def cargar_datos(ruta_csv):
    try:
        df = pd.read_csv(ruta_csv)
        assert "texto" in df.columns and "etiqueta" in df.columns, (
            "Faltan columnas 'texto' o 'etiqueta'"
        )
        assert not df["texto"].isna().any(), "Hay textos vacíos"
        assert df["etiqueta"].isin([0, 1]).all(), (
            "Etiquetas inválidas (solo 0 y 1 permitidos)"
        )
        return df
    except Exception as e:
        print("Error al cargar datos:", e)
        sys.exit(1)


# -------------------------------
# Función: entrenar modelo base
# -------------------------------
def entrenar_modelo(X_train, y_train):
    modelo = LogisticRegression(max_iter=100)
    modelo.fit(X_train, y_train)
    return modelo


# -------------------------------
# Función: mostrar matriz de confusión
# -------------------------------
def mostrar_matriz_confusion(y_true, y_pred):
    etiquetas = ["Real", "Fake"]
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusión:")
    print(cm)
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=etiquetas))


# -------------------------------
# Función: graficar curvas ROC y PR
# -------------------------------
def graficar_metricas(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(12, 5))

    # Curva ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Falsos Positivos")
    plt.ylabel("Verdaderos Positivos")
    plt.title("Curva ROC")
    plt.legend()

    # Curva Precision-Recall
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="orange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Función principal
# -------------------------------
def main(ruta_csv):
    df = cargar_datos(ruta_csv)
    print(f"Ejemplos disponibles: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["etiqueta"], test_size=0.2, random_state=42
    )

    vectorizador = CountVectorizer()
    X_train_vect = vectorizador.fit_transform(X_train)
    X_test_vect = vectorizador.transform(X_test)

    modelo = entrenar_modelo(X_train_vect, y_train)

    y_pred = modelo.predict(X_test_vect)
    y_proba = modelo.predict_proba(X_test_vect)[:, 1]  # Probabilidad de clase 1 (Fake)

    mostrar_matriz_confusion(y_test, y_pred)
    graficar_metricas(y_test, y_proba)


# -------------------------------
# Punto de entrada del script
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python modelo_logistico.py salida_limpia.csv")
        sys.exit(1)

    ruta = sys.argv[1]
    main(ruta)
