# 🧠 Ejercicio 131/200: Ensemble por Votación de Modelos Livianos en Clasificación Binaria
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.models import load_model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, TextVectorization
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)

# Configuración general
vocab_size = 5000  # Número de palabras únicas (reducido)
maxlen = 200  # Longitud máxima de cada secuencia
embedding_dim = 16  # Dimensión del embedding

# Carga el dataset IMDB (solo 5000 palabras más comunes)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Limita la cantidad de datos para entrenamiento rápido (opcional)
x_train = x_train[:8000]
y_train = y_train[:8000]
x_test = x_test[:2000]
y_test = y_test[:2000]

# Padding: asegura que todas las secuencias tengan la misma longitud
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


# Función para crear un modelo ligero
def crear_modelo():
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen
            ),
            GlobalAveragePooling1D(),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),  # Salida binaria
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# Entrena y guarda 3 modelos distintos con diferentes épocas
for i, epochs in enumerate([3, 4, 5], start=1):
    print(f"\nEntrenando modelo {i} con {epochs} épocas...")
    model = crear_modelo()
    model.fit(
        x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=2
    )
    model.save(f"modelo_{i}.h5")
    print(f"✅ Modelo modelo_{i}.h5 guardado correctamente.")

model_1 = load_model("modelo_1.h5")
model_2 = load_model("modelo_2.h5")
model_3 = load_model("modelo_3.h5")

fake = pd.read_csv(r"Datasets\archive\Fake.csv")  # Contiene 'text' y 'label'
true = pd.read_csv(r"Datasets\archive\True.csv")  # Contiene 'text' y 'label'
fake["label"] = 0
true["label"] = 1
data = pd.concat([fake, true], ignore_index=True)
texts = data["text"].tolist()
labels = data["label"].tolist()

# Crear y adaptar el vectorizador
vectorizer = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=maxlen
)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(32))

# Guardar el vectorizador como un modelo Keras funcional
vectorizer_model = Sequential([vectorizer])
vectorizer_model.save("vectorizer_layer.keras")

# Carga del vectorizador previamente guardado
vectorizer = load_model("vectorizer_layer.keras")

X_test = vectorizer(tf.constant(texts))

# Obtener probabilidades predichas por cada modelo
preds_1 = model_1.predict(X_test)
preds_2 = model_2.predict(X_test)
preds_3 = model_3.predict(X_test)

# Votación blanda: promedio de las probabilidades
ensemble_preds = (preds_1 + preds_2 + preds_3) / 3

# Clasificación final usando umbral 0.5
final_preds = (ensemble_preds > 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(labels, final_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Ensemble")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(labels, ensemble_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Ensemble")
plt.legend()
plt.show()

# Precision-Recall
precision, recall, _ = precision_recall_curve(labels, ensemble_preds)

plt.figure()
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Ensemble")
plt.legend()
plt.show()
