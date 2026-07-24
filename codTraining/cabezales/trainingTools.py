import torch
import copy
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import  TensorDataset, Subset, DataLoader


def train_eval_binary(
    DEVICE: str | torch.device,
    NUM_EPOCHS: int,
    clasificador: nn.Module,
    criterio,
    optimizador,
    train_loader,
    validation_loader,
    umbral: float = 0.5,
    patience: int = 5,
):
    """
    Entrena y valida un clasificador binario con early stopping.

    Calcula por época:

    - Loss de entrenamiento.
    - Accuracy de entrenamiento.
    - F1-score de entrenamiento.
    - Loss de validación.
    - Accuracy de validación.
    - F1-score de validación.

    El early stopping supervisa el F1 de validación.

    El clasificador debe producir un logit por observación:

        [batch_size, 1]

    El criterio recomendado es:

        nn.BCEWithLogitsLoss()

    Parámetros
    ----------
    DEVICE:
        Dispositivo de entrenamiento: "cpu", "cuda" o torch.device.

    NUM_EPOCHS:
        Número máximo de épocas.

    clasificador:
        Modelo de clasificación binaria.

    criterio:
        Función de pérdida.

    optimizador:
        Optimizador de PyTorch.

    train_loader:
        DataLoader de entrenamiento.

    validation_loader:
        DataLoader de validación.

    umbral:
        Umbral utilizado para convertir probabilidades en clases.

    patience:
        Número máximo de épocas consecutivas sin mejora en
        el F1 de validación antes de detener el entrenamiento.

    Retorna
    -------
    clasificador:
        Modelo con los pesos correspondientes al mejor F1
        de validación.

    resultados:
        DataFrame con las métricas de cada época ejecutada.
    """

    if NUM_EPOCHS <= 0:
        raise ValueError(
            "NUM_EPOCHS debe ser mayor que cero."
        )

    if patience <= 0:
        raise ValueError(
            "patience debe ser mayor que cero."
        )

    if not 0.0 <= umbral <= 1.0:
        raise ValueError(
            "umbral debe estar entre 0 y 1."
        )

    results = []

    # Mejor F1 de validación observado.
    best_val_f1 = float("-inf")

    # Copia de los pesos del mejor modelo.
    best_model_state = None

    # Época donde se encontró el mejor modelo.
    best_epoch = 0

    # Número de épocas consecutivas sin mejora.
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):

        print(f"\nEPOCH: {epoch + 1}")

        # ====================================================
        # ENTRENAMIENTO
        # ====================================================

        clasificador.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_true_labels = []
        train_predictions = []

        for vectores, etiquetas in train_loader:

            vectores = (
                vectores
                .to(DEVICE)
                .float()
            )

            etiquetas = (
                etiquetas
                .to(DEVICE)
                .float()
            )

            optimizador.zero_grad(
                set_to_none=True
            )

            # Forma esperada antes de squeeze:
            # [batch_size, 1]
            logits = clasificador(
                vectores
            ).squeeze(1)

            loss = criterio(
                logits,
                etiquetas,
            )

            loss.backward()

            optimizador.step()

            probabilidades = torch.sigmoid(
                logits
            )

            predicciones = (
                probabilidades >= umbral
            ).long()

            batch_size = etiquetas.size(0)

            train_loss += (
                loss.item() * batch_size
            )

            train_correct += (
                predicciones == etiquetas.long()
            ).sum().item()

            train_total += batch_size

            train_true_labels.extend(
                etiquetas
                .long()
                .detach()
                .cpu()
                .tolist()
            )

            train_predictions.extend(
                predicciones
                .detach()
                .cpu()
                .tolist()
            )

        if train_total == 0:
            raise ValueError(
                "El train_loader no contiene registros."
            )

        train_loss /= train_total

        train_accuracy = (
            train_correct / train_total
        )

        train_f1 = f1_score(
            train_true_labels,
            train_predictions,
            average="binary",
            pos_label=1,
            zero_division=0,
        )

        # ====================================================
        # VALIDACIÓN
        # ====================================================

        clasificador.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_true_labels = []
        val_predictions = []

        with torch.inference_mode():

            for vectores, etiquetas in validation_loader:

                vectores = (
                    vectores
                    .to(DEVICE)
                    .float()
                )

                etiquetas = (
                    etiquetas
                    .to(DEVICE)
                    .float()
                )

                logits = clasificador(
                    vectores
                ).squeeze(1)

                loss = criterio(
                    logits,
                    etiquetas,
                )

                probabilidades = torch.sigmoid(
                    logits
                )

                predicciones = (
                    probabilidades >= umbral
                ).long()

                batch_size = etiquetas.size(0)

                val_loss += (
                    loss.item() * batch_size
                )

                val_correct += (
                    predicciones == etiquetas.long()
                ).sum().item()

                val_total += batch_size

                val_true_labels.extend(
                    etiquetas
                    .long()
                    .cpu()
                    .tolist()
                )

                val_predictions.extend(
                    predicciones
                    .cpu()
                    .tolist()
                )

        if val_total == 0:
            raise ValueError(
                "El validation_loader no contiene registros."
            )

        val_loss /= val_total

        val_accuracy = (
            val_correct / val_total
        )

        val_f1 = f1_score(
            val_true_labels,
            val_predictions,
            average="binary",
            pos_label=1,
            zero_division=0,
        )

        # ====================================================
        # RESULTADOS DE LA ÉPOCA
        # ====================================================

        print(
            f"Época {epoch + 1}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
        }

        results.append(data)

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if val_f1 > best_val_f1:

            # Se encontró un modelo mejor.
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            # Reinicia el contador.
            epochs_without_improvement = 0

            # Guarda una copia independiente de los pesos.
            best_model_state = copy.deepcopy(
                clasificador.state_dict()
            )

            print(
                f"Mejora detectada | "
                f"Mejor Val F1: {best_val_f1:.4f}"
            )

        else:

            # No hubo mejora en esta época.
            epochs_without_improvement += 1

            print(
                "Épocas sin mejora: "
                f"{epochs_without_improvement}/{patience}"
            )

        # Detener cuando se alcanza patience.
        if epochs_without_improvement >= patience:

            print(
                "\nEarly stopping activado."
            )

            print(
                f"Mejor época: {best_epoch} | "
                f"Mejor Val F1: {best_val_f1:.4f}"
            )

            break

    # ========================================================
    # RESTAURAR EL MEJOR MODELO
    # ========================================================

    if best_model_state is not None:

        clasificador.load_state_dict(
            best_model_state
        )

        clasificador = clasificador.to(
            DEVICE
        )

    resultados = pd.DataFrame(
        results
    )

    return clasificador, resultados




def train_eval_multiclass(
    DEVICE: str | torch.device,
    NUM_EPOCHS: int,
    clasificador: nn.Module,
    criterio: nn.Module,
    optimizador: torch.optim.Optimizer,
    train_loader,
    validation_loader,
    patience: int = 5,
    min_delta: float = 0.0,
) -> tuple[nn.Module, pd.DataFrame]:
    """
    Entrena y valida un clasificador multiclase con early stopping.

    El clasificador debe retornar logits con forma:

        [batch_size, num_classes]

    Las etiquetas deben contener identificadores enteros:

        0, 1, 2, ..., num_classes - 1

    Se calculan las siguientes métricas por época:

        - Loss de entrenamiento.
        - Accuracy de entrenamiento.
        - Macro-F1 de entrenamiento.
        - Loss de validación.
        - Accuracy de validación.
        - Macro-F1 de validación.

    El early stopping supervisa el macro-F1 de validación.

    Parámetros
    ----------
    DEVICE:
        Dispositivo de ejecución:

            "cuda"
            "cpu"
            torch.device

    NUM_EPOCHS:
        Número máximo de épocas de entrenamiento.

    clasificador:
        Red neuronal multiclase.

    criterio:
        Función de pérdida.

        Normalmente:

            nn.CrossEntropyLoss()

    optimizador:
        Optimizador encargado de actualizar los parámetros.

    train_loader:
        DataLoader del conjunto de entrenamiento.

    validation_loader:
        DataLoader del conjunto de validación.

    patience:
        Número máximo de épocas consecutivas sin mejora
        significativa en el macro-F1 de validación.

    min_delta:
        Mejora mínima requerida para considerar que
        val_macro_f1 aumentó.

        Ejemplo:

            min_delta = 0.001

        significa que la mejora debe ser superior a 0.001.

    Retorna
    -------
    clasificador:
        Modelo con los pesos correspondientes al mejor
        macro-F1 de validación.

    resultados:
        DataFrame con las métricas de cada época ejecutada.
    """

    # ========================================================
    # VALIDACIONES
    # ========================================================

    if NUM_EPOCHS <= 0:
        raise ValueError(
            "NUM_EPOCHS debe ser mayor que cero."
        )

    if patience <= 0:
        raise ValueError(
            "patience debe ser mayor que cero."
        )

    if min_delta < 0:
        raise ValueError(
            "min_delta debe ser mayor o igual que cero."
        )

    # Almacena las métricas de cada época.
    results = []

    # Mejor macro-F1 observado en validación.
    best_val_macro_f1 = float("-inf")

    # Copia de los parámetros del mejor modelo.
    best_model_state = None

    # Época donde se obtuvo el mejor resultado.
    best_epoch = 0

    # Contador de épocas consecutivas sin mejora.
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):

        print(f"\nEPOCH: {epoch + 1}")

        # ====================================================
        # ENTRENAMIENTO
        # ====================================================

        clasificador.train()

        train_loss_sum = 0.0
        train_total = 0

        train_true_labels = []
        train_predictions = []

        for vectores, etiquetas in train_loader:

            # Los vectores de entrada deben ser float.
            vectores = (
                vectores
                .to(DEVICE)
                .float()
            )

            # CrossEntropyLoss requiere etiquetas long.
            etiquetas = (
                etiquetas
                .to(DEVICE)
                .long()
            )

            # Eliminar gradientes del lote anterior.
            optimizador.zero_grad(
                set_to_none=True
            )

            # Forma de salida:
            #
            # [batch_size, num_classes]
            logits = clasificador(
                vectores
            )

            # Calcular la pérdida multiclase.
            loss = criterio(
                logits,
                etiquetas,
            )

            # Retropropagación.
            loss.backward()

            # Actualizar parámetros.
            optimizador.step()

            # Clase con el mayor logit.
            predicciones = torch.argmax(
                logits,
                dim=1,
            )

            batch_size = etiquetas.size(0)

            # Acumular pérdida total.
            train_loss_sum += (
                loss.item() * batch_size
            )

            # Acumular número de ejemplos.
            train_total += batch_size

            # Guardar etiquetas reales.
            train_true_labels.extend(
                etiquetas
                .detach()
                .cpu()
                .tolist()
            )

            # Guardar predicciones.
            train_predictions.extend(
                predicciones
                .detach()
                .cpu()
                .tolist()
            )

        if train_total == 0:
            raise ValueError(
                "El train_loader no contiene registros."
            )

        # Pérdida promedio de entrenamiento.
        train_loss = (
            train_loss_sum / train_total
        )

        # Accuracy de entrenamiento.
        train_accuracy = accuracy_score(
            train_true_labels,
            train_predictions,
        )

        # Macro-F1 de entrenamiento.
        train_macro_f1 = f1_score(
            train_true_labels,
            train_predictions,
            average="macro",
            zero_division=0,
        )

        # ====================================================
        # VALIDACIÓN
        # ====================================================

        clasificador.eval()

        val_loss_sum = 0.0
        val_total = 0

        val_true_labels = []
        val_predictions = []

        with torch.inference_mode():

            for vectores, etiquetas in validation_loader:

                vectores = (
                    vectores
                    .to(DEVICE)
                    .float()
                )

                etiquetas = (
                    etiquetas
                    .to(DEVICE)
                    .long()
                )

                # Forma:
                #
                # [batch_size, num_classes]
                logits = clasificador(
                    vectores
                )

                # Pérdida de validación.
                loss = criterio(
                    logits,
                    etiquetas,
                )

                # Clase predicha.
                predicciones = torch.argmax(
                    logits,
                    dim=1,
                )

                batch_size = etiquetas.size(0)

                # Acumular pérdida total.
                val_loss_sum += (
                    loss.item() * batch_size
                )

                # Acumular registros.
                val_total += batch_size

                # Guardar etiquetas reales.
                val_true_labels.extend(
                    etiquetas
                    .cpu()
                    .tolist()
                )

                # Guardar predicciones.
                val_predictions.extend(
                    predicciones
                    .cpu()
                    .tolist()
                )

        if val_total == 0:
            raise ValueError(
                "El validation_loader no contiene registros."
            )

        # Pérdida promedio de validación.
        val_loss = (
            val_loss_sum / val_total
        )

        # Accuracy de validación.
        val_accuracy = accuracy_score(
            val_true_labels,
            val_predictions,
        )

        # Macro-F1 de validación.
        val_macro_f1 = f1_score(
            val_true_labels,
            val_predictions,
            average="macro",
            zero_division=0,
        )

        # ====================================================
        # RESULTADOS DE LA ÉPOCA
        # ====================================================

        print(
            f"Época {epoch + 1}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Train macro-F1: {train_macro_f1:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f} | "
            f"Val macro-F1: {val_macro_f1:.4f}"
        )

        results.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_macro_f1": train_macro_f1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
            }
        )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        # Se considera una mejora cuando supera al mejor
        # valor anterior por al menos min_delta.
        mejora = (
            val_macro_f1
            > best_val_macro_f1 + min_delta
        )

        if mejora:

            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch + 1

            # Reiniciar el contador.
            epochs_without_improvement = 0

            # Guardar una copia independiente del modelo.
            best_model_state = copy.deepcopy(
                clasificador.state_dict()
            )

            print(
                f"Mejora detectada | "
                f"Mejor Val macro-F1: "
                f"{best_val_macro_f1:.4f}"
            )

        else:

            epochs_without_improvement += 1

            print(
                "Épocas sin mejora: "
                f"{epochs_without_improvement}/{patience}"
            )

        # Detener cuando se alcance la paciencia.
        if epochs_without_improvement >= patience:

            print(
                "\nEarly stopping activado."
            )

            print(
                f"Mejor época: {best_epoch} | "
                f"Mejor Val macro-F1: "
                f"{best_val_macro_f1:.4f}"
            )

            break

    # ========================================================
    # RESTAURAR EL MEJOR MODELO
    # ========================================================

    if best_model_state is not None:

        clasificador.load_state_dict(
            best_model_state
        )

        clasificador = clasificador.to(
            DEVICE
        )

    resultados = pd.DataFrame(
        results
    )

    return clasificador, resultados



def makeDivision(tensor_dataset: TensorDataset, test_size:float, seed:int = 42):
    labels_numpy = tensor_dataset.tensors[1].cpu().numpy()
    idx =np.arange(len(tensor_dataset))
    train_indices, temp_indices=train_test_split(idx,test_size=test_size
                                             ,random_state=seed
                                             ,stratify=labels_numpy)
    test_indices, eval_indices=train_test_split(idx,test_size=0.50
                                             ,random_state=seed
                                             ,stratify=labels_numpy[temp_indices])
    train_dataset = Subset(
    tensor_dataset,
    train_indices,
)

    validation_dataset = Subset(
    tensor_dataset,
    eval_indices,
)

    test_dataset = Subset(
    tensor_dataset,
    test_indices,
)
    return train_dataset, test_dataset, validation_dataset

def  createLoaders(BATCH_SIZE,train_dataset,
                   validation_dataset,test_dataset):
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    return train_loader, validation_loader, test_loader

