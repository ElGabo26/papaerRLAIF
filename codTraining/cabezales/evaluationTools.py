import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)



def eval_binary(
    clasificador: nn.Module,
    loader,
    device: str | torch.device,
    criterio: nn.Module | None = None,
    umbral: float = 0.5,
) -> dict:
    """
    Evalúa un clasificador binario utilizando un DataLoader.

    El clasificador debe retornar logits con forma:

        [batch_size, 1]

    o:

        [batch_size]

    Las etiquetas deben contener valores:

        0 o 1

    Métricas calculadas
    -------------------
    - Loss promedio, si se proporciona criterio.
    - Accuracy.
    - Precision.
    - Recall o sensibilidad.
    - Especificidad.
    - F1-score.
    - ROC-AUC.
    - PR-AUC.
    - Matriz de confusión.
    - Informe por clase.

    Parámetros
    ----------
    clasificador:
        Modelo binario entrenado.

    loader:
        DataLoader de validación o prueba.

    device:
        Dispositivo de ejecución.

    criterio:
        Función de pérdida. Normalmente:

            nn.BCEWithLogitsLoss()

        Puede ser None si no se desea calcular loss.

    umbral:
        Umbral para convertir probabilidades en clases.

    Retorna
    -------
    dict:
        Diccionario con métricas, predicciones,
        probabilidades y etiquetas reales.
    """

    if not 0.0 <= umbral <= 1.0:
        raise ValueError(
            "El umbral debe encontrarse entre 0 y 1."
        )

    # Activar modo de evaluación.
    clasificador.eval()

    loss_total = 0.0
    total_registros = 0

    etiquetas_reales = []
    probabilidades_totales = []
    predicciones_totales = []

    with torch.inference_mode():

        for vectores, etiquetas in loader:

            # Mover vectores y etiquetas al dispositivo.
            vectores = vectores.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            etiquetas = etiquetas.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            # Obtener logits.
            logits = clasificador(vectores)

            # Convertir la salida a forma [batch_size].
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)

            elif logits.ndim != 1:
                raise ValueError(
                    "El clasificador binario debe retornar logits "
                    "con forma [batch_size] o [batch_size, 1]."
                )

            # Calcular pérdida, si se proporcionó criterio.
            if criterio is not None:

                loss = criterio(
                    logits,
                    etiquetas,
                )

                batch_size = etiquetas.size(0)

                loss_total += (
                    loss.item() * batch_size
                )

                total_registros += batch_size

            # Convertir logits en probabilidades.
            probabilidades = torch.sigmoid(
                logits
            )

            # Aplicar umbral.
            predicciones = (
                probabilidades >= umbral
            ).long()

            # Acumular tensores en CPU.
            etiquetas_reales.append(
                etiquetas
                .long()
                .detach()
                .cpu()
            )

            probabilidades_totales.append(
                probabilidades
                .detach()
                .cpu()
            )

            predicciones_totales.append(
                predicciones
                .detach()
                .cpu()
            )

    if len(etiquetas_reales) == 0:
        raise ValueError(
            "El DataLoader no contiene registros."
        )

    # Concatenar todos los lotes.
    y_true = torch.cat(
        etiquetas_reales
    ).numpy()

    y_prob = torch.cat(
        probabilidades_totales
    ).numpy()

    y_pred = torch.cat(
        predicciones_totales
    ).numpy()

    # Matriz de confusión con estructura fija:
    #
    # [[TN, FP],
    #  [FN, TP]]
    matriz_confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matriz_confusion.ravel()

    # Métricas principales.
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    # Especificidad:
    # proporción de negativos correctamente identificados.
    especificidad = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # ROC-AUC y PR-AUC requieren ambas clases.
    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )

        pr_auc = average_precision_score(
            y_true,
            y_prob,
        )

    else:

        roc_auc = np.nan
        pr_auc = np.nan

        print(
            "Advertencia: ROC-AUC y PR-AUC no pueden "
            "calcularse porque el loader contiene una sola clase."
        )

    # Loss promedio.
    if criterio is not None and total_registros > 0:

        loss_promedio = (
            loss_total / total_registros
        )

    else:

        loss_promedio = None

    # Informe detallado por clase.
    reporte_clasificacion = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Clase 0",
            "Clase 1",
        ],
        zero_division=0,
        output_dict=True,
    )

    resultados = {
        "loss": loss_promedio,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": especificidad,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": umbral,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": matriz_confusion,
        "classification_report": reporte_clasificacion,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    return resultados



def eval_multiclass(
    clasificador: nn.Module,
    loader,
    device: str | torch.device,
    criterio: nn.Module | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """
    Evalúa un clasificador multiclase utilizando un DataLoader.

    El clasificador debe retornar logits con forma:

        [batch_size, num_classes]

    Las etiquetas deben contener identificadores enteros:

        0, 1, 2, ..., num_classes - 1

    Métricas calculadas
    -------------------
    - Loss promedio, si se proporciona un criterio.
    - Accuracy.
    - Macro-precision.
    - Macro-recall.
    - Macro-F1.
    - Weighted-F1.
    - ROC-AUC multiclase, cuando puede calcularse.
    - Matriz de confusión.
    - Informe de clasificación por clase.

    Parámetros
    ----------
    clasificador:
        Modelo multiclase entrenado.

    loader:
        DataLoader de validación o prueba.

    device:
        Dispositivo de ejecución.

    criterio:
        Función de pérdida.

        Normalmente:

            nn.CrossEntropyLoss()

        Puede ser None si no se desea calcular loss.

    class_names:
        Nombres de las clases en el mismo orden que sus índices.

        Ejemplo:

            [
                "addition",
                "subtraction",
                "geometry",
            ]

        Si es None, se generan nombres como:

            Clase 0, Clase 1, ...

    Retorna
    -------
    dict:
        Diccionario con métricas, matriz de confusión,
        reporte por clase, predicciones, probabilidades
        y etiquetas reales.
    """

    # Activa el modo de evaluación.
    clasificador.eval()

    loss_total = 0.0
    total_registros = 0

    etiquetas_reales = []
    predicciones_totales = []
    probabilidades_totales = []

    with torch.inference_mode():

        for vectores, etiquetas in loader:

            # Mover vectores a CPU o GPU.
            vectores = vectores.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            # CrossEntropyLoss requiere etiquetas enteras long.
            etiquetas = etiquetas.to(
                device,
                dtype=torch.long,
                non_blocking=True,
            )
            print(torch.unique(etiquetas))
            # Forma esperada:
            #
            # [batch_size, num_classes]
            logits = clasificador(vectores)

            if logits.ndim != 2:
                raise ValueError(
                    "El clasificador multiclase debe retornar "
                    "logits con forma [batch_size, num_classes]."
                )

            if logits.shape[0] != etiquetas.shape[0]:
                raise ValueError(
                    "La cantidad de logits y etiquetas no coincide."
                )

            # Calcular pérdida, si se proporcionó criterio.
            if criterio is not None:

                loss = criterio(
                    logits,
                    etiquetas,
                )

                batch_size = etiquetas.size(0)

                loss_total += (
                    loss.item() * batch_size
                )

                total_registros += batch_size

            # Convertir logits en probabilidades multiclase.
            probabilidades = torch.softmax(
                logits,
                dim=1,
            )

            # Seleccionar la clase con el logit más alto.
            predicciones = torch.argmax(
                logits,
                dim=1,
            )

            # Guardar resultados en CPU.
            etiquetas_reales.append(
                etiquetas
                .detach()
                .cpu()
            )

            predicciones_totales.append(
                predicciones
                .detach()
                .cpu()
            )

            probabilidades_totales.append(
                probabilidades
                .detach()
                .cpu()
            )

    if len(etiquetas_reales) == 0:
        raise ValueError(
            "El DataLoader no contiene registros."
        )

    # Concatenar los resultados de todos los lotes.
    y_true = torch.cat(
        etiquetas_reales
    ).numpy()

    y_pred = torch.cat(
        predicciones_totales
    ).numpy()

    y_prob = torch.cat(
        probabilidades_totales
    ).numpy()

    # Número de clases definido por la salida del modelo.
    num_classes = y_prob.shape[1]
    print("num_clases CALCULADO")

    # Comprobar o generar nombres de clases.
    if class_names is None:

        class_names = [
            f"Clase {i}"
            for i in range(num_classes)
        ]

    elif len(class_names) != num_classes:

        raise ValueError(
            "La cantidad de nombres en class_names debe "
            "coincidir con num_classes."
        )

    labels = list(
        range(num_classes)
    )

    # ========================================================
    # MÉTRICAS PRINCIPALES
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="micro",
        zero_division=0,
    )

    # ========================================================
    # MATRIZ DE CONFUSIÓN
    # ========================================================

    matriz_confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    # ========================================================
    # ROC-AUC MULTICLASE
    # ========================================================

    clases_presentes = np.unique(
        y_true
    )

    if len(clases_presentes) == num_classes:

        try:

            roc_auc_macro_ovr = roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )

            roc_auc_weighted_ovr = roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )

        except ValueError:

            roc_auc_macro_ovr = np.nan
            roc_auc_weighted_ovr = np.nan

    else:

        roc_auc_macro_ovr = np.nan
        roc_auc_weighted_ovr = np.nan

        print(
            "Advertencia: no se calculó ROC-AUC porque "
            "el loader no contiene ejemplos de todas las clases."
        )

    # ========================================================
    # LOSS PROMEDIO
    # ========================================================

    if criterio is not None and total_registros > 0:

        loss_promedio = (
            loss_total / total_registros
        )

    else:

        loss_promedio = None

    # ========================================================
    # REPORTE POR CLASE
    # ========================================================

    reporte_clasificacion = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    resultados = {
        "loss": loss_promedio,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "roc_auc_macro_ovr": roc_auc_macro_ovr,
        "roc_auc_weighted_ovr": roc_auc_weighted_ovr,
        "num_classes": num_classes,
        "class_names": class_names,
        "confusion_matrix": matriz_confusion,
        "classification_report": reporte_clasificacion,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    return resultados

