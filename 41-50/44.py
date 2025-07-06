# 🧠 Ejercicio 44/200: Clasificación binaria de texto usando LSTM (Spam vs Ham)
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, LSTM, Dense
from keras.optimizers import Adam

# 1️⃣ Cargar el dataset y convertir etiquetas a binario
df = pd.read_csv(
    "Datasets/sms+spam+collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"],
)
df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})

# 🔍 Verificar balance de clases
sns.countplot(x=df["label"])
plt.title("Distribución de clases")
plt.show()

# 2️⃣ Vectorización del texto
vectorizer = TextVectorization(
    max_tokens=10000, output_mode="int", output_sequence_length=50
)
vectorizer.adapt(df["text"])

# 3️⃣ Convertir texto a secuencia de enteros
X = vectorizer(tf.constant(df["text"])).numpy()
y = df["label_bin"].values

# 4️⃣ Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5️⃣ Construcción del modelo con LSTM
model = Sequential(
    [
        Embedding(input_dim=10000, output_dim=64),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),  # Dropout para regularizar
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# 6️⃣ Entrenamiento
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,  # Mayor batch mejora el gradiente en secuencias
    validation_split=0.2,
    verbose=1,
)

# 7️⃣ Evaluación
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.4).astype(int).flatten()

print("\n📋 Reporte de Clasificación:")
print(
    classification_report(y_test, y_pred, target_names=["Ham", "Spam"], zero_division=0)
)

# 8️⃣ Matriz de Confusión
conf = confusion_matrix(y_test, y_pred)
sns.heatmap(
    conf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()
