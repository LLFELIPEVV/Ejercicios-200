# 🧠 Ejercicio 42/200: Clasificación binaria con Keras - Detección de spam en SMS
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import TextVectorization, Dense, Flatten, Embedding

# 1️⃣ Cargar dataset (descargado localmente desde UCI - SMS Spam Collection)
archivo = r"Datasets\sms+spam+collection\SMSSpamCollection"
df = pd.read_csv(archivo, sep="\t", header=None, names=["label", "text"])

# 2️⃣ Convertir etiquetas categóricas a binarias: ham → 0, spam → 1
df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})

# 3️⃣ Crear capa de vectorización de texto (similar a Tokenizer de scikit-learn)
vectorize_layer = TextVectorization(
    max_tokens=1000,  # Limita el tamaño del vocabulario
    output_mode="int",  # Codifica texto como secuencias de enteros
    output_sequence_length=10,  # Longitud fija de cada secuencia (relleno o truncamiento)
)

# Adaptar el vectorizador al corpus
vectorize_layer.adapt(df["text"])

# Visualizar el vocabulario aprendido (opcional)
vocabulario = vectorize_layer.get_vocabulary()
print(f"📚 Vocabulario aprendido (primeros 20 tokens):\n{vocabulario[:20]}")

# Vectorizar el texto
X = vectorize_layer(tf.constant(df["text"])).numpy()
y = df["label_bin"].values

# 4️⃣ División de datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5️⃣ Construcción del modelo secuencial (MLP + embeddings)
model = Sequential(
    [
        Embedding(
            input_dim=1000, output_dim=8
        ),  # Capa de embeddings (vector semántico por token)
        Flatten(),  # Aplanar salida de embeddings para pasar a densas
        Dense(16, activation="relu"),  # Capa oculta con activación ReLU
        Dense(1, activation="sigmoid"),  # Capa de salida (probabilidad binaria)
    ]
)

# 6️⃣ Compilación del modelo
model.compile(
    optimizer=Adam(),  # Optimizador Adam (buena elección por defecto)
    loss="binary_crossentropy",  # Pérdida para clasificación binaria
    metrics=["accuracy"],
)

# 7️⃣ Entrenamiento del modelo
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,  # Validación interna
    epochs=15,
    batch_size=2,
    verbose=1,
)

# 8️⃣ Evaluación y métricas
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Reporte de clasificación
print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# Matriz de confusión
conf = confusion_matrix(y_test, y_pred)
sns.heatmap(
    conf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.title("Matriz de Confusión - Clasificación Binaria")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.tight_layout()
plt.show()
