# 🧠 Ejercicio 46/200: Clasificación binaria de texto usando GRU (Spam vs Ham)
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, GRU, Dense
from keras.optimizers import Adam

# 1️⃣ Cargar el dataset de mensajes y etiquetar como binario
df = pd.read_csv(
    "Datasets/sms+spam+collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"],
)
df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})

# 2️⃣ Vectorización del texto a secuencias de enteros
vectorizer = TextVectorization(
    max_tokens=10000,  # Límite del vocabulario
    output_sequence_length=50,  # Longitud fija de cada mensaje
    output_mode="int",
)
vectorizer.adapt(df["text"])  # Aprende el vocabulario a partir del texto

# 3️⃣ Aplicar la vectorización
X = vectorizer(tf.constant(df["text"])).numpy()
y = df["label_bin"].values

# 4️⃣ División del dataset (80% train - 20% test), respetando la proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 5️⃣ Definición del modelo secuencial con GRU
model = Sequential(
    [
        Embedding(input_dim=10000, output_dim=64),  # Representación semántica densa
        GRU(64),  # Captura el contexto secuencial
        Dense(1, activation="sigmoid"),  # Clasificación binaria
    ]
)

# 6️⃣ Compilar el modelo
model.compile(
    optimizer=Adam(),  # Optimizador recomendado para texto
    loss="binary_crossentropy",  # Función de pérdida para clasificación binaria
    metrics=["accuracy"],
)

# 7️⃣ Entrenar el modelo
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,  # Parte del set de entrenamiento se reserva para validación
    epochs=10,
    batch_size=4,
    verbose=1,
)

# 8️⃣ Predecir y evaluar
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Umbral de 0.5 para clasificar

# 9️⃣ Mostrar reporte de clasificación
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# 🔟 Visualizar la matriz de confusión
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.title("Matriz de Confusión - GRU")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# 1️⃣1️⃣ Visualizar métricas del entrenamiento
plt.figure(figsize=(12, 5))

# Curva de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Pérdida entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida validación")
plt.title("Curva de pérdida")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()

# Curva de precisión
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Precisión entrenamiento")
plt.plot(history.history["val_accuracy"], label="Precisión validación")
plt.title("Curva de precisión")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
