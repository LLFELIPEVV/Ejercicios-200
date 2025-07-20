# 🧠 Ejercicio 166/200 — Ensemble por Votación: Scikit-learn + Keras
import sys
import pandas as pd
import keras_tuner as kt
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


# -----------------------
# Función: validación básica
# -----------------------
def cargar_y_validar(ruta_csv):
    try:
        df = pd.read_csv(ruta_csv)
        assert "texto" in df.columns and "etiqueta" in df.columns, (
            "Columnas requeridas no encontradas"
        )
        assert df["etiqueta"].isin([0, 1]).all(), "Etiquetas inválidas detectadas"
        assert not df["texto"].isna().any(), "Textos vacíos en el dataset"
        return df
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


# -----------------------
# Función: gráfico de métricas
# -----------------------
def graficar_metricas(y_true, y_proba, titulo="Modelo"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.title(f"ROC - {titulo}")
    plt.xlabel("Falsos Positivos")
    plt.ylabel("Verdaderos Positivos")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="orange")
    plt.title(f"Precision-Recall - {titulo}")
    plt.xlabel("Recall")
    plt.ylabel("Precisión")

    plt.tight_layout()
    plt.show()


# -----------------------
# Modelo Keras simple con tuner
# -----------------------
def construir_modelo(hp):
    model = Sequential()
    model.add(
        Dense(
            units=hp.Choice("units", [16, 32]),
            activation="relu",
            input_shape=(input_dim,),
        )
    )
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice("lr", [1e-2, 1e-3])),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -----------------------
# Función principal
# -----------------------
def main(ruta_csv):
    global input_dim  # Requerido por el tuner
    df = cargar_y_validar(ruta_csv)

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["etiqueta"], test_size=0.2, random_state=42
    )

    vectorizer = CountVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    input_dim = X_train_vect.shape[1]

    # -------- Modelo 1: Regresión Logística --------
    modelo_rl = LogisticRegression(max_iter=1000)
    modelo_rl.fit(X_train_vect, y_train)
    proba_rl = modelo_rl.predict_proba(X_test_vect)[:, 1]

    # -------- Modelo 2: Red Neuronal con KerasTuner --------
    tuner = kt.RandomSearch(
        construir_modelo,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=1,
        directory="tuner_ensemble",
        project_name="keras_simple",
    )
    tuner.search(
        X_train_vect.toarray(), y_train, epochs=10, validation_split=0.2, verbose=0
    )
    modelo_keras = tuner.get_best_models(1)[0]
    proba_keras = modelo_keras.predict(X_test_vect.toarray()).flatten()

    # -------- Ensemble: Votación Blanda --------
    proba_ensemble = (proba_rl + proba_keras) / 2

    # -------- Evaluaciones --------
    for nombre, proba in zip(
        ["Regresión Logística", "Red Neuronal Keras", "Ensemble"],
        [proba_rl, proba_keras, proba_ensemble],
    ):
        print(f"\n===== {nombre} =====")
        pred = (proba >= 0.5).astype(int)
        print("Matriz de Confusión:")
        print(confusion_matrix(y_test, pred))
        print("Reporte:")
        print(classification_report(y_test, pred, target_names=["Real", "Fake"]))
        graficar_metricas(y_test, proba, titulo=nombre)


# -----------------------
# Punto de entrada
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ensemble_votacion.py salida_limpia.csv")
        sys.exit(1)

    main(sys.argv[1])
