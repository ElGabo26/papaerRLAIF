from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer


# ============================================================
# 1. Configuración
# ============================================================



# ============================================================
# 2. Cargar encoder y tokenizer
# ============================================================

def load_encoder(
    encoder_path: str,
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path
    )

    encoder = AutoModel.from_pretrained(
        encoder_path
    )

    encoder.to(device)
    encoder.eval()

    return tokenizer, encoder


# ============================================================
# 3. Cargar el cabezal completo
# ============================================================

def load_complete_classifier(
    classifier_path: Path,
    device: torch.device,
) -> nn.Module:

    if not classifier_path.exists():
        raise FileNotFoundError(
            f"No se encontró el cabezal: {classifier_path}"
        )

    classifier = torch.load(
        classifier_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(classifier, nn.Module):
        raise TypeError(
            "El archivo no contiene un modelo completo "
            "de tipo torch.nn.Module."
        )

    classifier.to(device)
    classifier.eval()

    return classifier


# ============================================================
# 4. Cargar LabelEncoder
# ============================================================

def load_label_encoder(
    labels_path: Path,
) -> LabelEncoder:

    if not labels_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {labels_path}"
        )

    label_encoder=joblib.load(labels_path)

    return label_encoder


# ============================================================
# 5. Mean pooling
# ============================================================

def mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calcula el promedio de los embeddings de los tokens válidos.

    Parámetros
    ----------
    last_hidden_state:
        Tensor con forma:
        [batch_size, sequence_length, hidden_size]

    attention_mask:
        Tensor con forma:
        [batch_size, sequence_length]

        Contiene:
        - 1 para tokens reales.
        - 0 para tokens de padding.

    Retorna
    -------
    torch.Tensor:
        Tensor con forma:
        [batch_size, hidden_size]
    """

    # Convierte:
    # [batch_size, sequence_length]
    #
    # en:
    # [batch_size, sequence_length, 1]
    expanded_mask = attention_mask.unsqueeze(-1)

    # Adapta el tipo de dato de la máscara al tipo
    # de los embeddings, por ejemplo float16 o bfloat16.
    expanded_mask = expanded_mask.to(
        dtype=last_hidden_state.dtype
    )

    # Los embeddings de los tokens de padding se multiplican por 0.
    masked_embeddings = (
        last_hidden_state * expanded_mask
    )

    # Suma los embeddings de los tokens válidos.
    summed_embeddings = masked_embeddings.sum(
        dim=1
    )

    # Cuenta cuántos tokens válidos tiene cada texto.
    valid_token_count = expanded_mask.sum(
        dim=1
    )

    # Evita una división por cero.
    valid_token_count = valid_token_count.clamp(
        min=1e-9
    )

    # Promedio de los tokens válidos.
    pooled_embeddings = (
        summed_embeddings / valid_token_count
    )

    return pooled_embeddings


# ============================================================
# 6. Crear embeddings mediante mean pooling
# ============================================================

@torch.inference_mode()
def create_embeddings(
    prompts: Sequence[str],
    tokenizer,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 128,
) -> torch.Tensor:

    if len(prompts) == 0:
        raise ValueError(
            "La colección de prompts está vacía."
        )

    all_embeddings = []

    for start in range(0, len(prompts), batch_size):

        batch_prompts = prompts[
            start:start + batch_size
        ]

        encoded = tokenizer(
            list(batch_prompts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        outputs = encoder(**encoded)

        # Forma:
        # [batch_size, sequence_length, hidden_size]
        last_hidden_state = outputs.last_hidden_state

        # Mean pooling considerando únicamente tokens reales.
        embeddings = mean_pooling(
            last_hidden_state=last_hidden_state,
            attention_mask=encoded["attention_mask"],
        )

        # Mueve los embeddings a CPU para reducir el uso
        # acumulado de memoria de la GPU.
        all_embeddings.append(
            embeddings.cpu()
        )

    return torch.cat(
        all_embeddings,
        dim=0,
    )


# ============================================================
# 7. Clasificar prompts
# ============================================================
@torch.inference_mode()
def classify_prompts(
    prompts,
    tokenizer,
    encoder,
    classifier,
    label_encoder,
    device,
    batch_size=8,
    max_length=128,
    threshold=0.5,
):
    embeddings = create_embeddings(
        prompts=prompts,
        tokenizer=tokenizer,
        encoder=encoder,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    predictions = []

    for start in range(0, len(embeddings), batch_size):

        batch_embeddings = embeddings[
            start:start + batch_size
        ].to(device)

        logits = classifier(batch_embeddings)

        # ----------------------------------------------------
        # Clasificación binaria:
        # salida [batch_size] o [batch_size, 1]
        # ----------------------------------------------------
        if logits.ndim == 1 or logits.shape[1] == 1:

            logits = logits.squeeze(-1)

            positive_probabilities = torch.sigmoid(logits)

            predicted_indices = (
                positive_probabilities >= threshold
            ).long()

            positive_probabilities = (
                positive_probabilities
                .detach()
                .cpu()
                .numpy()
            )

            predicted_indices = (
                predicted_indices
                .detach()
                .cpu()
                .numpy()
            )

            for local_index, predicted_index in enumerate(
                predicted_indices
            ):
                prompt_index = start + local_index

                positive_probability = float(
                    positive_probabilities[local_index]
                )

                negative_probability = (
                    1.0 - positive_probability
                )

                predicted_label = label_encoder.inverse_transform(
                    [int(predicted_index)]
                )[0]

                confidence = max(
                    negative_probability,
                    positive_probability,
                )

                row = {
                    "response": prompts[prompt_index],
                    "predicted_index": int(predicted_index),
                    "predicted_label": str(predicted_label),
                    "confidence": float(confidence),
                    "probability_class_0": negative_probability,
                    "probability_class_1": positive_probability,
                }

                # Añadir los nombres reales de las clases.
                if len(label_encoder.classes_) == 2:
                    row[
                        f"prob_{label_encoder.classes_[0]}"
                    ] = negative_probability

                    row[
                        f"prob_{label_encoder.classes_[1]}"
                    ] = positive_probability

                predictions.append(row)

        # ----------------------------------------------------
        # Clasificación multiclase:
        # salida [batch_size, num_classes]
        # ----------------------------------------------------
        else:

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_indices = probabilities.argmax(
                dim=1
            )

            confidence_values = probabilities.max(
                dim=1
            ).values

            probabilities = (
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

            predicted_indices = (
                predicted_indices
                .detach()
                .cpu()
                .numpy()
            )

            confidence_values = (
                confidence_values
                .detach()
                .cpu()
                .numpy()
            )

            for local_index, predicted_index in enumerate(
                predicted_indices
            ):
                prompt_index = start + local_index

                predicted_label = label_encoder.inverse_transform(
                    [int(predicted_index)]
                )[0]

                row = {
                    "response": prompts[prompt_index],
                    "predicted_index": int(predicted_index),
                    "predicted_label": str(predicted_label),
                    "confidence": float(
                        confidence_values[local_index]
                    ),
                }

                for class_position, class_name in enumerate(
                    label_encoder.classes_
                ):
                    row[f"prob_{class_name}"] = float(
                        probabilities[
                            local_index,
                            class_position,
                        ]
                    )

                predictions.append(row)

    return pd.DataFrame(predictions)

# ============================================================
# 8. Ejecución principal
# ============================================================

def evaluateResponses(BASE_RESPONSES:str, responseColumn:str,ENCODER_PATH:str,CLASSIFIER_PATH:str, LABELS_PATH:str,BATCH_SIZE:int, MAX_LENGTH:int,DEVICE:str) -> pd.DataFrame:

    responseBase= pd.read_csv(f"{BASE_RESPONSES}")
    responses=responseBase[responseColumn].to_list()
    print("Cargando tokenizer y encoder...")

    tokenizer, encoder = load_encoder(
        encoder_path=ENCODER_PATH,
        device=DEVICE,
    )

    print(
        f"Dimensión de los embeddings: "
        f"{encoder.config.hidden_size}"
    )

    print("Cargando cabezal clasificador...")

    classifier = load_complete_classifier(
        classifier_path=CLASSIFIER_PATH,
        device=DEVICE,
    )

    print("Cabezal cargado correctamente:")
    print(classifier)

    print("Cargando etiquetas...")

    label_encoder = load_label_encoder(
        labels_path=LABELS_PATH
    )

    print(
        "Etiquetas:",
        list(label_encoder.classes_),
    )

    print("Clasificando prompts...")

    results = classify_prompts(
        prompts=responses,
        tokenizer=tokenizer,
        encoder=encoder,
        classifier=classifier,
        label_encoder=label_encoder,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    columns_to_show = [
        "response",
        "predicted_index",
        "predicted_label",
        "confidence",
    ]

    print("\nResultados:")
    

    return results
