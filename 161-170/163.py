# 🧠 Ejercicio 163/200 — Ensemble por Votación con Modelos Simples (Light VotingClassifier)
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)


# -------------------------------
# Función: cargar y validar datos
# -------------------------------
def cargar_datos(ruta_csv):
    try:
        df = pd.read_csv(ruta_csv)
        assert "texto" in df.columns and "etiqueta" in df.columns, (
            "Faltan columnas necesarias"
        )
        assert not df["texto"].isna().any(), "Textos vacíos detectados"
        assert df["etiqueta"].isin([0, 1]).all(), (
            "Etiquetas no válidas (deben ser 0 o 1)"
        )
        return df
    except Exception as e:
        print("Error al cargar datos:", e)
        sys.exit(1)


# -------------------------------
# Función: graficar métricas
# -------------------------------
def graficar_metricas(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(12, 5))

    # Curva ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("Curva ROC")
    plt.xlabel("Falsos Positivos")
    plt.ylabel("Verdaderos Positivos")
    plt.legend()

    # Curva Precision-Recall
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="orange")
    plt.title("Curva Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precisión")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Función: mostrar resultados
# -------------------------------
def mostrar_resultados(y_true, y_pred):
    etiquetas = ["Real", "Fake"]
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusión:")
    print(cm)
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=etiquetas))


# -------------------------------
# Función principal
# -------------------------------
def main(ruta_csv):
    df = cargar_datos(ruta_csv)
    print(f"Ejemplos cargados: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["etiqueta"], test_size=0.2, random_state=42
    )

    vectorizador = CountVectorizer()
    X_train_vect = vectorizador.fit_transform(X_train)
    X_test_vect = vectorizador.transform(X_test)

    # Clasificadores base
    clf1 = LogisticRegression(max_iter=100)
    clf2 = MultinomialNB()
    clf3 = CalibratedClassifierCV(LinearSVC(max_iter=100), cv=3)

    # Ensemble por votación
    ensemble = VotingClassifier(
        estimators=[("lr", clf1), ("nb", clf2), ("svc", clf3)],
        voting="soft",  # Promedio de probabilidades
    )

    # Entrenar ensemble
    ensemble.fit(X_train_vect, y_train)

    # Predicciones
    y_pred = ensemble.predict(X_test_vect)
    y_proba = ensemble.predict_proba(X_test_vect)[:, 1]

    # Mostrar métricas
    mostrar_resultados(y_test, y_pred)
    graficar_metricas(y_test, y_proba)


# -------------------------------
# Punto de entrada
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ensemble_votacion.py salida_limpia.csv")
        sys.exit(1)

    ruta = sys.argv[1]
    main(ruta)
