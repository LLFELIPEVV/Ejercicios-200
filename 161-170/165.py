# 🧠 Ejercicio 165/200 — Red Neuronal Simple en Keras + KerasTuner
import sys
import pandas as pd
import keras_tuner as kt
import matplotlib.pyplot as plt

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
        assert df["etiqueta"].isin([0, 1]).all(), "Etiquetas no válidas (solo 0 o 1)"
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

    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("Curva ROC")
    plt.xlabel("Falsos Positivos")
    plt.ylabel("Verdaderos Positivos")
    plt.legend()

    # PR
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
# Definición del modelo Keras para KerasTuner
# -------------------------------
def construir_modelo(hp):
    model = Sequential()
    # Número de neuronas tunable: 16, 32 o 64
    model.add(
        Dense(
            units=hp.Choice("units", [16, 32, 64]),
            activation="relu",
            input_shape=(input_dim,),
        )
    )
    model.add(Dense(1, activation="sigmoid"))

    # Learning rate tunable
    lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------------
# Función principal
# -------------------------------
def main(ruta_csv):
    global input_dim  # requerido para que `construir_modelo` acceda al shape

    df = cargar_datos(ruta_csv)
    print(f"Ejemplos cargados: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["etiqueta"], test_size=0.2, random_state=42
    )

    vectorizador = CountVectorizer()
    X_train_vect = vectorizador.fit_transform(X_train)
    X_test_vect = vectorizador.transform(X_test)

    input_dim = X_train_vect.shape[1]  # para definir entrada

    # Crear tuner
    tuner = kt.RandomSearch(
        construir_modelo,
        objective="val_accuracy",
        max_trials=4,
        executions_per_trial=1,
        directory="tuner_logs",
        project_name="fake_news_keras",
    )

    tuner.search(
        X_train_vect.toarray(), y_train, epochs=10, validation_split=0.2, verbose=1
    )

    # Obtener mejor modelo
    mejor_modelo = tuner.get_best_models(num_models=1)[0]

    # Evaluar
    y_proba = mejor_modelo.predict(X_test_vect.toarray()).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    mostrar_resultados(y_test, y_pred)
    graficar_metricas(y_test, y_proba)


# -------------------------------
# Punto de entrada del script
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python keras_nn_tuning.py salida_limpia.csv")
        sys.exit(1)

    ruta = sys.argv[1]
    main(ruta)
