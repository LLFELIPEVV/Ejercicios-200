# 🧪 Ejercicio 83/200 — Clasificación de Fake News con PyTorch (Red Neuronal Densa)
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import gc  # Para manejo de memoria
import psutil  # Para monitoreo de sistema

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# ⚙️ Configuración CONSERVADORA para CPU limitada
# ------------------------------------------------------------

# Configuración específica para Ryzen 3 2200U (2 núcleos físicos, 4 hilos)
torch.set_num_threads(2)  # Solo 2 hilos para evitar sobrecargar el sistema
torch.set_num_interop_threads(1)  # Un solo hilo para operaciones entre operadores

# Configuración muy conservadora para evitar bloqueos
BATCH_SIZE = 16  # Reducido significativamente
NUM_WORKERS = 0  # Sin workers adicionales para evitar sobrecarga
PIN_MEMORY = False  # Desactivado para ahorrar memoria

# Configuración de memoria
torch.backends.cudnn.benchmark = False  # Desactivar optimizaciones que consumen memoria

print(
    f"🖥️ Sistema: {psutil.cpu_count()} núcleos lógicos, {psutil.virtual_memory().total // (1024**3)} GB RAM"
)
print(f"⚙️ PyTorch configurado para {torch.get_num_threads()} hilos")

# ------------------------------------------------------------
# 📥 1. Carga y preparación de datos (con manejo de memoria)
# ------------------------------------------------------------

print("📥 Cargando datos...")
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

# Combinar y limpiar datos
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()

# Liberar memoria inmediatamente
del df_fake, df_true
gc.collect()

print(f"📊 Dataset: {len(df)} muestras")

X, y = df["text"].values, df["label"].values
del df  # Liberar DataFrame original
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"🔄 Split: {len(X_train)} entrenamiento, {len(X_test)} prueba")

# ------------------------------------------------------------
# 🔠 2. Vectorización con TF-IDF (configuración conservadora)
# ------------------------------------------------------------

print("🔠 Vectorizando texto...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000,  # Reducido para menor uso de memoria
    min_df=2,  # Filtrar palabras muy raras
    max_df=0.95,  # Filtrar palabras muy comunes
)

X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Liberar memoria
del X, X_train, X_test
gc.collect()

print(f"📝 Vocabulario: {X_train_vec.shape[1]} características")
print(f"💾 Memoria actual: {psutil.virtual_memory().percent:.1f}% usada")

# ------------------------------------------------------------
# 📦 3. Dataset personalizado con manejo eficiente de memoria
# ------------------------------------------------------------


class FakeNewsDataset(Dataset):
    def __init__(self, features, labels):
        # Convertir a tensores de forma eficiente
        self.X = torch.from_numpy(features).float()
        self.y = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Crear datasets
train_dataset = FakeNewsDataset(X_train_vec, y_train)
test_dataset = FakeNewsDataset(X_test_vec, y_test)

# Liberar arrays numpy
del X_train_vec, X_test_vec, y_train, y_test
gc.collect()

# DataLoaders con configuración conservadora
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=True,  # Evitar batches incompletos
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

# ------------------------------------------------------------
# 🧠 4. Modelo más simple y eficiente
# ------------------------------------------------------------


class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleFeedForwardNN, self).__init__()
        # Arquitectura mucho más simple para CPU limitada
        self.fc1 = nn.Linear(input_dim, 64)  # Reducido de 128 a 64
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 16)  # Reducido de 32 a 16
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return torch.sigmoid(self.output(x))


# Forzar uso de CPU
device = torch.device("cpu")
model = SimpleFeedForwardNN(input_dim=train_dataset.X.shape[1]).to(device)

print(f"🧠 Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")

# ------------------------------------------------------------
# ⚙️ 5. Configuración de entrenamiento conservadora
# ------------------------------------------------------------

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
EPOCHS = 5  # Reducido para pruebas iniciales

# ------------------------------------------------------------
# 🏋️ 6. Entrenamiento con monitoreo de memoria
# ------------------------------------------------------------

print("🏋️ Iniciando entrenamiento...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    batch_count = 0

    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        # Monitorear memoria cada 10 batches
        if batch_idx % 10 == 0:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:  # Si se usa más del 85% de RAM
                print(f"⚠️ Memoria alta: {memory_percent:.1f}% - Ejecutando limpieza...")
                gc.collect()

        batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        # Mostrar progreso cada 50 batches
        if batch_idx % 50 == 0:
            print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / batch_count
    memory_percent = psutil.virtual_memory().percent
    print(
        f"🔁 Epoch {epoch + 1}/{EPOCHS} — Loss promedio: {avg_loss:.4f}, Memoria: {memory_percent:.1f}%"
    )

    # Limpiar memoria entre épocas
    gc.collect()

# ------------------------------------------------------------
# 📊 7. Evaluación del modelo
# ------------------------------------------------------------

print("📊 Evaluando modelo...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        preds = (outputs.cpu().numpy() >= 0.5).astype(int).flatten()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

print("\n📈 Reporte de clasificación (PyTorch Optimizado):")
print(classification_report(all_labels, all_preds, target_names=["Fake", "Real"]))
print("\n🧾 Matriz de confusión:")
print(confusion_matrix(all_labels, all_preds))

print(f"\n💾 Uso final de memoria: {psutil.virtual_memory().percent:.1f}%")
print("✅ Entrenamiento completado sin bloqueos!")
