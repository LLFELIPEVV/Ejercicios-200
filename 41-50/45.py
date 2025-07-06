# 🧠 Ejercicio 45/200: Clasificación binaria de texto con BiLSTM (Spam vs Ham)
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense
from keras.optimizers import Adam

# 1️⃣ Cargar dataset y preparar etiquetas binarizadas
df = pd.read_csv(
    "Datasets/sms+spam+collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"],
)
df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})  # Mapear a 0 (ham) y 1 (spam)

# 2️⃣ Preprocesamiento: vectorizar texto
vectorizer = TextVectorization(
    max_tokens=10000,  # Limita vocabulario a 10,000 palabras más frecuentes
    output_mode="int",  # Convierte tokens en enteros
    output_sequence_length=50,  # Longitud fija de las secuencias
)
vectorizer.adapt(df["text"])  # Aprende el vocabulario del dataset

# 3️⃣ Transformar texto a secuencias de enteros y extraer etiquetas
X = vectorizer(tf.constant(df["text"])).numpy()
y = df["label_bin"].values

# 4️⃣ Dividir en entrenamiento y testeo manteniendo proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5️⃣ Definir el modelo BiLSTM
model = Sequential(
    [
        Embedding(
            input_dim=10000, output_dim=64
        ),  # Embedding para representar semánticamente cada palabra
        Bidirectional(
            LSTM(64)
        ),  # LSTM bidireccional para entender contexto hacia adelante y atrás
        Dense(1, activation="sigmoid"),  # Capa de salida binaria (spam = 1, ham = 0)
    ]
)

# 6️⃣ Compilar el modelo
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# 7️⃣ Entrenar el modelo
history = model.fit(
    X_train, y_train, validation_split=0.2, epochs=10, batch_size=4, verbose=1
)

# 8️⃣ Evaluación del modelo en el conjunto de prueba
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Umbral de clasificación binaria

# 9️⃣ Reporte de métricas
print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# 🔟 Matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.title("Matriz de Confusión - BiLSTM")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
