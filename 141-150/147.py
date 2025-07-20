# 🧠 Ejercicio 147/200: Ensemble liviano por votación de modelos simples con validación profesional en consola
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential, load_model
from keras.layers import Dense, TextVectorization

# 1. Cargar y preparar datos
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")

df_fake["label"] = 0
df_true["label"] = 1
df = pd.concat([df_fake, df_true], ignore_index=True)

# Limpieza básica
df = df.dropna(subset=["text", "label"])
X_texts = df["text"].astype(str).tolist()
y = df["label"].values

# 2. Tokenización básica
tokenizer = TextVectorization(max_tokens=1000, output_mode="int")
tokenizer.adapt(X_texts)
X_pad = tokenizer(np.array(X_texts)).numpy()

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)


# 3. Definición de modelo simple
def crear_modelo(seed):
    np.random.seed(seed)
    model = Sequential(
        [
            Dense(16, activation="relu", input_shape=(X_pad.shape[1],)),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# 4. Entrenar 3 modelos simples con distinta semilla
for i in range(1, 4):
    model = crear_modelo(seed=i)
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    model.save(f"model_{i}.h5")  # Guardar modelo

# 5. Cargar modelos y hacer predicciones individuales
predicciones = []
for i in range(1, 4):
    assert os.path.exists(f"model_{i}.h5"), f"Modelo model_{i}.h5 no encontrado"
    model = load_model(f"model_{i}.h5")
    y_prob = model.predict(X_test, verbose=0).flatten()
    predicciones.append(y_prob)

# 6. Votación por promedio → luego redondeo
y_promedio = np.mean(predicciones, axis=0)
y_pred = np.round(y_promedio)  # 0 o 1

# 7. Validación profesional
print("=== Matriz de Confusión ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Reporte de Clasificación ===")
print(classification_report(y_test, y_pred, digits=3))

# 8. Validación automática con assert
assert len(y_pred) == len(y_test), (
    "¡Las predicciones y etiquetas tienen diferente longitud!"
)
assert set(np.unique(y_pred)).issubset({0.0, 1.0}), "Las predicciones no son binarias"
print("✅ Validaciones superadas: predicción correcta y binaria.")
