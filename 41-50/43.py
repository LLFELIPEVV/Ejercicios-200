# 🧠 Ejercicio 43/200: Clasificación binaria de texto con Embedding + GlobalAveragePooling1D (Spam vs Ham)
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense

# 1️⃣ Cargar y preparar el dataset SMS Spam Collection
archivo = r"Datasets\sms+spam+collection\SMSSpamCollection"
df = pd.read_csv(archivo, sep="\t", header=None, names=["label", "text"])

# 2️⃣ Mapear etiquetas de texto ('ham', 'spam') a valores binarios (0, 1)
df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})

# 3️⃣ Vectorización de texto usando TextVectorization
# - Convierte texto crudo a secuencias de enteros (tokens)
# - Limita el vocabulario a 1000 palabras y secuencias a 20 tokens
vectorize_layer = TextVectorization(
    max_tokens=1000, output_mode="int", output_sequence_length=20
)
vectorize_layer.adapt(df["text"])  # Aprende el vocabulario

# 4️⃣ Transformar texto a secuencias de enteros
X = vectorize_layer(tf.constant(df["text"])).numpy()
y = df["label_bin"].values

# 5️⃣ Dividir el dataset en entrenamiento y testeo (80/20), manteniendo la proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6️⃣ Definir el modelo con Embedding y GlobalAveragePooling1D
model = Sequential(
    [
        Embedding(input_dim=1000, output_dim=16),  # Capa de representación semántica
        GlobalAveragePooling1D(),  # Promedia los vectores (agrega información de toda la secuencia)
        Dense(16, activation="relu"),  # Capa densa intermedia
        Dense(1, activation="sigmoid"),  # Salida binaria (spam vs no spam)
    ]
)

# 7️⃣ Compilar el modelo
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# 8️⃣ Entrenamiento del modelo
history = model.fit(
    X_train, y_train, epochs=15, batch_size=4, validation_split=0.2, verbose=1
)

# 9️⃣ Predicción en el conjunto de testeo
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# 🔟 Evaluación del modelo
target_names = ["Ham", "Spam"]
print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 🔍 Matriz de confusión
conf = confusion_matrix(y_test, y_pred)
sns.heatmap(
    conf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
