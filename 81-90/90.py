# 🧪 Ejercicio 90/200 — Comparación práctica Keras vs PyTorch: Clasificación de Fake News con TF-IDF + Red Densa
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras import backend as K

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------------------
# ⚙️ Configuración optimizada para hardware limitado
# ----------------------------------------------------
# Configurar uso de memoria GPU para TensorFlow
if tf.config.experimental.list_physical_devices("GPU"):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Configurar threads para CPU de 4 núcleos
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Batch size pequeño para 8GB RAM
batch_size = 32

# ----------------------------------------------------
# 📥 Carga y vectorización de datos con optimización de memoria
# ----------------------------------------------------
print("📥 Cargando datos...")

# Cargar datasets con chunks para evitar sobrecarga de memoria
df_fake = pd.read_csv("Datasets/archive/Fake.csv", usecols=["text"])
df_fake["label"] = 0
df_true = pd.read_csv("Datasets/archive/True.csv", usecols=["text"])
df_true["label"] = 1

# Concatenar y limpiar inmediatamente
df = pd.concat([df_fake, df_true], ignore_index=True)
del df_fake, df_true  # Liberar memoria inmediatamente
gc.collect()

# Filtrar datos nulos y muestrear para reducir carga
df = df.dropna().sample(frac=0.7, random_state=42).reset_index(drop=True)
print(f"📊 Datos después del muestreo: {len(df)} registros")

X, y = df["text"].values, df["label"].values
del df
gc.collect()

# Split de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Vectorización con menos features para ahorrar memoria
print("🔄 Vectorizando texto...")
vectorizer = TfidfVectorizer(
    max_features=2000,  # Reducido de 5000 a 2000
    stop_words="english",
    max_df=0.8,  # Ignorar términos muy frecuentes
    min_df=3,  # Ignorar términos muy raros
    dtype=np.float32,  # Usar float32 en lugar de float64
)

# Convertir matrices sparse a dense solo cuando sea necesario
X_train_sparse = vectorizer.fit_transform(X_train)
X_test_sparse = vectorizer.transform(X_test)

# Liberar memoria de texto original
del X, X_train, X_test
gc.collect()

# ----------------------------------------------------
# 🔶 Modelo con Keras (TF) - Optimizado
# ----------------------------------------------------
print("🔶 Creando modelo Keras optimizado...")

# Convertir a dense solo para entrenamiento
X_train_vec = X_train_sparse.toarray().astype(np.float32)
# No eliminar X_train_sparse aún, se necesita para PyTorch
gc.collect()

# Modelo más pequeño para hardware limitado
model_keras = Sequential(
    [
        Input(shape=(2000,)),
        Dense(16, activation="relu"),  # Reducido de 32 a 16 neuronas
        Dropout(0.3),  # Aumentado dropout para evitar overfitting
        Dense(1, activation="sigmoid"),
    ]
)

model_keras.compile(
    optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"]
)

print("📌 Entrenando modelo Keras...")
history_keras = model_keras.fit(
    X_train_vec,
    y_train,
    epochs=5,  # Más epochs pero modelo más pequeño
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1,
)

# Evaluación Keras
print("📊 Evaluando modelo Keras...")
X_test_vec = X_test_sparse.toarray().astype(np.float32)
y_pred_keras = (
    (model_keras.predict(X_test_vec, batch_size=batch_size) > 0.5).astype(int).flatten()
)

print("\n📊 Clasificación (Keras):")
print(
    classification_report(
        y_test, y_pred_keras, target_names=["Fake", "Real"], zero_division=0
    )
)

# Limpieza intermedia - mantener X_train_sparse para PyTorch
del X_train_vec
gc.collect()
K.clear_session()

# ----------------------------------------------------
# 🔷 Modelo con PyTorch - Optimizado
# ----------------------------------------------------
print("🔷 Creando modelo PyTorch optimizado...")


class OptimizedFFNN(nn.Module):
    def __init__(self):
        super(OptimizedFFNN, self).__init__()
        self.fc1 = nn.Linear(2000, 16)  # Reducido de 32 a 16
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return torch.sigmoid(self.fc2(x))


# Preparación de datos con procesamiento por lotes
print("🔄 Preparando datos para PyTorch...")

# Convertir solo lo necesario para PyTorch
X_train_torch = torch.tensor(X_train_sparse.toarray(), dtype=torch.float32)
y_train_torch = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_torch = torch.tensor(X_test_vec, dtype=torch.float32)
y_test_torch = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# Limpieza - ahora sí eliminar las matrices sparse
del X_train_sparse, X_test_sparse, X_test_vec
gc.collect()

# DataLoader optimizado
train_ds = TensorDataset(X_train_torch, y_train_torch)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

# Inicializar modelo
model_torch = OptimizedFFNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model_torch.parameters(), lr=1e-3)

print("📌 Entrenando modelo PyTorch...")
model_torch.train()

for epoch in range(5):
    epoch_loss = 0
    for batch_idx, (xb, yb) in enumerate(train_dl):
        pred = model_torch(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Limpieza periódica de gradientes
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Epoch {epoch + 1}/5, Loss: {epoch_loss / len(train_dl):.4f}")

# Evaluación PyTorch
print("📊 Evaluando modelo PyTorch...")
model_torch.eval()
with torch.no_grad():
    y_pred_torch = (model_torch(X_test_torch) > 0.5).int().numpy().flatten()

print("\n📊 Clasificación (PyTorch):")
print(
    classification_report(
        y_test, y_pred_torch, target_names=["Fake", "Real"], zero_division=0
    )
)

# ----------------------------------------------------
# 📊 Comparativa visual optimizada
# ----------------------------------------------------
print("📊 Generando comparativa visual...")

labels = ["Keras", "PyTorch"]
accuracies = [np.mean(y_pred_keras == y_test), np.mean(y_pred_torch == y_test)]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, accuracies, color=["orange", "blue"], alpha=0.7)
plt.title(
    "Comparación de precisión (TF-IDF + Red Densa)\n"
)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Agregar valores en las barras
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{acc:.3f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()

# ----------------------------------------------------
# ♻️ Limpieza final agresiva
# ----------------------------------------------------
print("♻️ Limpiando memoria...")

# Limpiar variables grandes
del X_train_torch, y_train_torch, X_test_torch, y_test_torch
del train_ds, train_dl
del model_keras, model_torch
del y_pred_keras, y_pred_torch
del vectorizer

# Forzar limpieza de memoria
gc.collect()
K.clear_session()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("✅ Optimización completada. Memoria liberada.")
