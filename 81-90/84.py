# 🧪 Ejercicio 84/200 — Comparación entre Keras y PyTorch: Clasificación de Fake News con Red Densa
import os
import gc
import numpy as np
import pandas as pd
import warnings
from contextlib import contextmanager
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Suprimir warnings para salida más limpia
warnings.filterwarnings("ignore")


# --------------------------------------------------
# ⚙️ Configuración Global Optimizada
# --------------------------------------------------
class Config:
    def __init__(self):
        self.num_threads = min(
            os.cpu_count(), 8
        )  # Limitar threads para evitar sobrecarga
        self.batch_size = self._get_optimal_batch_size()
        self.epochs = 3
        self.max_features = 5000
        self.test_size = 0.2
        self.random_state = 42
        self.learning_rate = 1e-3

    def _get_optimal_batch_size(self):
        if self.num_threads <= 4:
            return 32
        elif self.num_threads <= 8:
            return 64
        else:
            return 128


config = Config()


# --------------------------------------------------
# 📊 Carga y Preprocesamiento del Dataset
# --------------------------------------------------
def load_and_preprocess_data():
    """Carga y preprocesa el dataset de fake news"""
    print("📊 Cargando dataset...")

    try:
        df_fake = pd.read_csv("Datasets/archive/Fake.csv")
        df_true = pd.read_csv("Datasets/archive/True.csv")

        df_fake["label"] = 0
        df_true["label"] = 1
        df = pd.concat([df_fake, df_true])[["text", "label"]].dropna()

        print(f"Dataset cargado: {len(df)} muestras")
        print(f"Distribución - Fake: {len(df_fake)}, Real: {len(df_true)}")

        X, y = df["text"].values, df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            stratify=y,
            random_state=config.random_state,
        )

        return X_train, X_test, y_train, y_test

    except FileNotFoundError:
        print("❌ Error: No se encontraron los archivos del dataset")
        print("Asegúrate de que los archivos estén en 'Datasets/archive/'")
        return None, None, None, None


# --------------------------------------------------
# 🧹 Contexto para Keras (TensorFlow)
# --------------------------------------------------
@contextmanager
def keras_context():
    """Context manager para configurar y limpiar recursos de Keras/TensorFlow"""
    try:
        # Configurar TensorFlow al inicio
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["OMP_NUM_THREADS"] = str(config.num_threads)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(config.num_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = "2"

        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(config.num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)

        # Configurar memoria GPU si está disponible
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error configurando GPU: {e}")

        yield tf

    finally:
        # Limpiar recursos de TensorFlow
        if "tf" in locals():
            tf.keras.backend.clear_session()
        gc.collect()


def run_keras_model(X_train, X_test, y_train, y_test):
    """Entrena y evalúa el modelo de Keras"""
    print("\n🧠 Iniciando entrenamiento con Keras...")

    with keras_context():
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Input
        from keras.optimizers import Adam

        # Vectorización TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )

        X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = vectorizer.transform(X_test).toarray()

        # Definición del modelo
        model = Sequential(
            [
                Input(shape=(config.max_features,)),
                Dense(64, activation="relu"),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dropout(0.1),
                Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        print("Arquitectura del modelo Keras:")
        model.summary()

        # Entrenamiento
        history = model.fit(
            X_train_tfidf,
            y_train,
            validation_split=0.1,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=1,
        )

        # Evaluación
        y_pred_proba = model.predict(X_test_tfidf, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Limpiar memoria
        del X_train_tfidf, X_test_tfidf, model, history
        gc.collect()

        return y_pred, y_test


# --------------------------------------------------
# 🔥 Contexto para PyTorch
# --------------------------------------------------
@contextmanager
def pytorch_context():
    """Context manager para configurar y limpiar recursos de PyTorch"""
    try:
        import torch

        # Configurar PyTorch
        torch.set_num_threads(config.num_threads)
        torch.set_num_interop_threads(2)

        # Configurar device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        yield torch, device

    finally:
        # Limpiar recursos de PyTorch
        if "torch" in locals():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()


def run_pytorch_model(X_train, X_test, y_train, y_test):
    """Entrena y evalúa el modelo de PyTorch"""
    print("\n🔥 Iniciando entrenamiento con PyTorch...")

    with pytorch_context() as (torch, device):
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset

        # Vectorización TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )

        X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = vectorizer.transform(X_test).toarray()

        # Dataset personalizado
        class FakeNewsDataset(Dataset):
            def __init__(self, features, labels):
                self.X = torch.tensor(features, dtype=torch.float32)
                self.y = torch.tensor(labels, dtype=torch.float32)

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        # Modelo mejorado
        class OptimizedFeedForwardNN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.dropout1 = nn.Dropout(0.3)
                self.fc2 = nn.Linear(64, 32)
                self.dropout2 = nn.Dropout(0.2)
                self.fc3 = nn.Linear(32, 16)
                self.dropout3 = nn.Dropout(0.1)
                self.out = nn.Linear(16, 1)

                # Inicialización de pesos
                self._init_weights()

            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.constant_(m.bias, 0)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout1(x)
                x = F.relu(self.fc2(x))
                x = self.dropout2(x)
                x = F.relu(self.fc3(x))
                x = self.dropout3(x)
                return torch.sigmoid(self.out(x))

        # Crear datasets y dataloaders
        train_dataset = FakeNewsDataset(X_train_tfidf, y_train)
        test_dataset = FakeNewsDataset(X_test_tfidf, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Evitar problemas de multiprocessing
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, num_workers=0
        )

        # Inicialización del modelo
        model = OptimizedFeedForwardNN(train_dataset.X.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )

        print(
            f"Modelo PyTorch creado con {sum(p.numel() for p in model.parameters())} parámetros"
        )

        # Entrenamiento
        model.train()
        for epoch in range(config.epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}/{config.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                    )

            accuracy = 100 * correct / total
            print(
                f"Epoch {epoch + 1}/{config.epochs} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )

        # Evaluación
        model.eval()
        preds, true_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                batch_preds = (outputs.cpu().numpy() >= 0.5).astype(int).flatten()
                preds.extend(batch_preds)
                true_labels.extend(y_batch.numpy())

        # Limpiar memoria
        del X_train_tfidf, X_test_tfidf, model, train_dataset, test_dataset
        del train_loader, test_loader
        gc.collect()

        return np.array(preds), np.array(true_labels)


# --------------------------------------------------
# 📊 Función Principal de Comparación
# --------------------------------------------------
def compare_frameworks():
    """Función principal que ejecuta la comparación entre frameworks"""
    print("🚀 Iniciando comparación entre Keras y PyTorch")
    print("=" * 60)

    # Cargar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    if X_train is None:
        return

    results = {}

    # Ejecutar Keras
    try:
        y_pred_keras, y_true_keras = run_keras_model(X_train, X_test, y_train, y_test)
        results["keras"] = {
            "predictions": y_pred_keras,
            "true_labels": y_true_keras,
            "accuracy": accuracy_score(y_true_keras, y_pred_keras),
        }
        print("✅ Keras completado exitosamente")

    except Exception as e:
        print(f"❌ Error en Keras: {e}")
        results["keras"] = None

    # Forzar limpieza de memoria entre frameworks
    gc.collect()

    # Ejecutar PyTorch
    try:
        y_pred_pytorch, y_true_pytorch = run_pytorch_model(
            X_train, X_test, y_train, y_test
        )
        results["pytorch"] = {
            "predictions": y_pred_pytorch,
            "true_labels": y_true_pytorch,
            "accuracy": accuracy_score(y_true_pytorch, y_pred_pytorch),
        }
        print("✅ PyTorch completado exitosamente")

    except Exception as e:
        print(f"❌ Error en PyTorch: {e}")
        results["pytorch"] = None

    # Mostrar resultados
    print_comparison_results(results)


def print_comparison_results(results):
    """Imprime los resultados de la comparación"""
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DE LA COMPARACIÓN")
    print("=" * 60)

    if results["keras"] is not None:
        print("\n🧠 === REPORTE KERAS ===")
        print(f"Accuracy: {results['keras']['accuracy']:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                results["keras"]["true_labels"],
                results["keras"]["predictions"],
                target_names=["Fake", "Real"],
            )
        )
        print("\nConfusion Matrix:")
        print(
            confusion_matrix(
                results["keras"]["true_labels"], results["keras"]["predictions"]
            )
        )

    if results["pytorch"] is not None:
        print("\n🔥 === REPORTE PYTORCH ===")
        print(f"Accuracy: {results['pytorch']['accuracy']:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                results["pytorch"]["true_labels"],
                results["pytorch"]["predictions"],
                target_names=["Fake", "Real"],
            )
        )
        print("\nConfusion Matrix:")
        print(
            confusion_matrix(
                results["pytorch"]["true_labels"], results["pytorch"]["predictions"]
            )
        )

    # Comparación final
    if results["keras"] is not None and results["pytorch"] is not None:
        print("\n🏆 === COMPARACIÓN FINAL ===")
        print(f"Keras Accuracy:   {results['keras']['accuracy']:.4f}")
        print(f"PyTorch Accuracy: {results['pytorch']['accuracy']:.4f}")

        if results["keras"]["accuracy"] > results["pytorch"]["accuracy"]:
            print("🥇 Keras obtuvo mejor accuracy")
        elif results["pytorch"]["accuracy"] > results["keras"]["accuracy"]:
            print("🥇 PyTorch obtuvo mejor accuracy")
        else:
            print("🤝 Ambos frameworks obtuvieron la misma accuracy")

    print("\n✅ Comparación completada!")


# --------------------------------------------------
# 🚀 Ejecución Principal
# --------------------------------------------------
if __name__ == "__main__":
    compare_frameworks()
