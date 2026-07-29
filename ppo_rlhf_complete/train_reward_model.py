from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from trl import RewardConfig, RewardTrainer


REQUIRED_COLUMNS = {"prompt", "chosen", "rejected"}

PREFERENCES_CSV="/workspace/papaerRLAIF/codTraining/calify/preferencesQwen.csv"
MODEL_PATH="/workspace/models/Qwen2.5-1.5B-Instruct"
OUTPUT_MODEL_PATH="/workspace/adaptedModels/PPO/rewardModels/Qwen2.5-1.5B-Instruct"
OUTPUT_MERGED_MODEL_PATH="/workspace/adaptedModels/PPO/mergedRewardModels/Qwen2.5-1.5B-Instruct"

# ============================================================
# CONFIGURACIÓN DEL REWARD MODEL
# ============================================================
#
# Los parámetros están agrupados por función. Los marcados como PRIORITARIOS
# son los primeros que conviene variar en un diseño experimental.

CONFIG: dict[str, Any] = {
    "paths": {
        # Modelo base Instruct/SFT. Debe pertenecer a la misma familia y usar
        # el mismo tokenizer que la política que después se entrenará con PPO.
        "model": MODEL_PATH,

        # CSV de preferencias con las columnas prompt, chosen y rejected.
        "preferences": PREFERENCES_CSV,

        # Directorio de checkpoints y del adaptador LoRA del Reward Model.
        "output_dir": OUTPUT_MODEL_PATH,

        # Directorio del Reward Model autocontenido que utilizará PPO.
        "merged_output_dir": OUTPUT_MERGED_MODEL_PATH,
    },
    "data": {
        # Fracción del dataset reservada para validación.
        "test_size": 0.15,

        # Procesos de CPU utilizados para preparar los pares de preferencias.
        "num_proc": 4,

        # Elimina pares repetidos para evitar que reciban mayor ponderación.
        "remove_duplicates": True,
    },
    "training": {
        # PRIORITARIO. Recorridos completos sobre todos los datos de entrenamiento.
        # Se mantienen las tres épocas del código original.
        "num_train_epochs": 3.0,

        # PRIORITARIO. Magnitud de la actualización de los adaptadores LoRA.
        "learning_rate": 5e-5,

        # Prompts preferidos/rechazados procesados simultáneamente por GPU.
        # El valor 2 mejora el uso de una GPU de 24 GB. Si aparece CUDA OOM,
        # debe reducirse a 1 y duplicar gradient_accumulation_steps.
        "per_device_train_batch_size": 4,

        # Batch utilizado durante la evaluación. Como no calcula gradientes,
        # normalmente puede ser mayor que el batch de entrenamiento.
        "per_device_eval_batch_size": 2,

        # Acumula gradientes antes de actualizar los pesos.
        # Con una GPU, batch efectivo = 2 × 4 = 8 preferencias.
        "gradient_accumulation_steps": 4,

        # PRIORITARIO. Longitud máxima de prompt + respuesta.
        # No reduce el dataset; solo recorta secuencias que superan este límite.
        "max_length": 512,

        # Regularización L2 aplicada por AdamW.
        "weight_decay": 0.05,

        # Fracción inicial para aumentar progresivamente el learning rate.
        "warmup_ratio": 0.03,

        # Evolución del learning rate después del calentamiento.
        "lr_scheduler_type": "cosine",

        # Norma máxima de los gradientes para evitar actualizaciones inestables.
        "max_grad_norm": 1.0,

        # PRIORITARIO. Penaliza recompensas desplazadas lejos de cero.
        # Facilita que la escala del Reward Model sea estable.
        "center_rewards_coefficient": 1e-2,

        # Semilla para división de datos, inicialización y muestreo.
        "seed": 42,
    },
    "lora": {
        # PRIORITARIO. Rango/capacidad de las matrices LoRA.
        "r": 8,

        # Factor de escalamiento de la actualización LoRA.
        "alpha": 32,

        # Regularización aplicada a los adaptadores LoRA.
        "dropout": 0.10,

        # No entrena los términos bias del modelo base.
        "bias": "none",

        # Aplica LoRA a las capas lineales del Transformer.
        "target_modules": "all-linear",

        # Conserva el cabezal escalar que asigna la recompensa.
        "modules_to_save": ["score"],
    },
    "performance": {
        # Implementación optimizada de atención incluida en PyTorch.
        "attention_implementation": "sdpa",

        # TensorFloat-32 acelera operaciones FP32 en GPUs Ampere o posteriores.
        # No modifica BF16 y normalmente no cambia la calidad estadística.
        "allow_tf32": True,

        # False es más rápido porque evita recalcular activaciones.
        # Cambiar a True únicamente cuando el modelo no cabe en VRAM.
        "gradient_checkpointing": False,

        # Optimizador AdamW con kernels fusionados de PyTorch.
        "optim": "adamw_torch_fused",

        # Se deja desactivado para evitar tiempos de compilación y problemas
        # con modelos que usan rutas dinámicas.
        "torch_compile": False,

        # Reduce el pico de RAM durante la carga del modelo.
        "low_cpu_mem_usage": True,

        # Trabajadores usados por los DataLoader.
        "dataloader_num_workers": 6,

        # Acelera transferencias CPU → GPU.
        "dataloader_pin_memory": True,

        # Conserva los trabajadores entre épocas.
        "dataloader_persistent_workers": True,

        # Agrupa secuencias de longitudes parecidas para reducir padding.
        "group_by_length": True,
    },
    "logging": {
        # Evalúa y registra métricas al terminar cada época.
        "eval_strategy": "epoch",
        "logging_strategy": "epoch",

        # Guarda un checkpoint al terminar cada época.
        "save_strategy": "epoch",

        # Conserva solamente los dos checkpoints más recientes/mejores.
        "save_total_limit": 2,

        # Recupera automáticamente el checkpoint con mayor eval_accuracy.
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,

        # TensorBoard permite inspeccionar la evolución interactiva.
        "report_to": "tensorboard",

        # Dataset principal: configuración y métricas por época.
        "epoch_metrics_filename": "reward_metrics_by_epoch.csv",

        # Configuración completa efectivamente utilizada.
        "config_filename": "reward_experiment_config.json",
    },
}


PRIORITY_CONFIG_COLUMNS = {
    "config_model": ("paths", "model"),
    "config_learning_rate": ("training", "learning_rate"),
    "config_num_train_epochs": ("training", "num_train_epochs"),
    "config_max_length": ("training", "max_length"),
    "config_train_batch_size": ("training", "per_device_train_batch_size"),
    "config_eval_batch_size": ("training", "per_device_eval_batch_size"),
    "config_gradient_accumulation": (
        "training",
        "gradient_accumulation_steps",
    ),
    "config_weight_decay": ("training", "weight_decay"),
    "config_warmup_ratio": ("training", "warmup_ratio"),
    "config_max_grad_norm": ("training", "max_grad_norm"),
    "config_center_rewards_coefficient": (
        "training",
        "center_rewards_coefficient",
    ),
    "config_lora_r": ("lora", "r"),
    "config_lora_alpha": ("lora", "alpha"),
    "config_lora_dropout": ("lora", "dropout"),
    "config_seed": ("training", "seed"),
}


def nested_value(config: dict[str, Any], path: tuple[str, str]) -> Any:
    """Obtiene un valor desde una ruta de dos niveles del diccionario."""
    return config[path[0]][path[1]]


def select_dtype() -> tuple[torch.dtype, bool, bool]:
    """
    Selecciona la precisión de entrenamiento.

    BF16 es la primera opción en GPUs compatibles porque mantiene el rango
    dinámico de FP32 y suele ser más estable que FP16.
    """
    if not torch.cuda.is_available():
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def supports_tf32() -> bool:
    """Indica si la GPU tiene capacidad Ampere (8.0) o posterior."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def configure_performance(config: dict[str, Any]) -> None:
    """Activa optimizaciones que no eliminan datos ni épocas."""
    if not torch.cuda.is_available():
        return

    allow_tf32 = config["performance"]["allow_tf32"] and supports_tf32()
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def load_preference_dataset(
    csv_path: str,
    data_config: dict[str, Any],
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Carga, valida, limpia y divide los pares chosen/rejected."""
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en el CSV: {sorted(missing)}. "
            f"Se requieren: {sorted(REQUIRED_COLUMNS)}"
        )

    ordered_columns = ["prompt", "chosen", "rejected"]
    frame = frame[ordered_columns].dropna().copy()

    for column in ordered_columns:
        frame[column] = frame[column].astype(str).str.strip()

    frame = frame[
        (frame["prompt"] != "")
        & (frame["chosen"] != "")
        & (frame["rejected"] != "")
        & (frame["chosen"] != frame["rejected"])
    ]

    if data_config["remove_duplicates"]:
        frame = frame.drop_duplicates(subset=ordered_columns)

    frame = frame.reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("Se necesitan al menos dos preferencias válidas.")

    dataset = Dataset.from_pandas(frame, preserve_index=False)

    def to_conversation(example: dict[str, str]) -> dict[str, list[dict[str, str]]]:
        """
        Convierte cada fila al formato conversacional de RewardTrainer.

        El mismo prompt antecede tanto a chosen como a rejected para que la
        comparación mida exclusivamente la preferencia entre respuestas.
        """
        return {
            "chosen": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["chosen"]},
            ],
            "rejected": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["rejected"]},
            ],
        }

    dataset = dataset.map(
        to_conversation,
        num_proc=data_config["num_proc"],
        remove_columns=dataset.column_names,
        desc="Construyendo pares de preferencias",
    )

    split = dataset.train_test_split(
        test_size=data_config["test_size"],
        seed=seed,
    )
    return split["train"], split["test"]


def build_epoch_metrics_dataset(
    log_history: list[dict[str, Any]],
    config: dict[str, Any],
    train_dataset_size: int,
    eval_dataset_size: int,
    run_id: str,
) -> pd.DataFrame:
    """
    Combina los logs de entrenamiento y validación en una fila por época.

    RewardTrainer registra la pérdida de entrenamiento y, por separado, las
    métricas ``eval_*``. Esta función las une mediante el campo ``epoch``.
    """
    history = pd.DataFrame(log_history)
    if history.empty or "epoch" not in history.columns:
        raise RuntimeError(
            "No se encontraron métricas por época en TrainerState.log_history."
        )

    history = history[history["epoch"].notna()].copy()
    history["dataset_epoch"] = history["epoch"].apply(
        lambda value: max(1, int(math.ceil(float(value) - 1e-9)))
    )
    history.insert(0, "run_id", run_id)

    completed_epochs = int(math.ceil(config["training"]["num_train_epochs"]))
    world_size = int(torch.distributed.get_world_size()) if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ) else 1

    priority_config = {
        name: nested_value(config, path)
        for name, path in PRIORITY_CONFIG_COLUMNS.items()
    }
    priority_config["config_effective_batch_size"] = (
        config["training"]["per_device_train_batch_size"]
        * config["training"]["gradient_accumulation_steps"]
        * world_size
    )
    priority_config["train_dataset_size"] = train_dataset_size
    priority_config["eval_dataset_size"] = eval_dataset_size

    rows: list[dict[str, Any]] = []
    for epoch_number in range(1, completed_epochs + 1):
        epoch_logs = history[history["dataset_epoch"] == epoch_number]
        if epoch_logs.empty:
            continue

        row: dict[str, Any] = {
            "run_id": run_id,
            "dataset_epoch": epoch_number,
        }

        # Los logs sin prefijo eval_ corresponden al entrenamiento.
        if "loss" in epoch_logs.columns:
            values = pd.to_numeric(epoch_logs["loss"], errors="coerce").dropna()
            if not values.empty:
                row["train_loss"] = float(values.mean())

        if "grad_norm" in epoch_logs.columns:
            values = pd.to_numeric(
                epoch_logs["grad_norm"],
                errors="coerce",
            ).dropna()
            if not values.empty:
                row["train_grad_norm_mean"] = float(values.mean())
                row["train_grad_norm_max"] = float(values.max())

        if "learning_rate" in epoch_logs.columns:
            values = pd.to_numeric(
                epoch_logs["learning_rate"],
                errors="coerce",
            ).dropna()
            if not values.empty:
                row["learning_rate_end"] = float(values.iloc[-1])

        # Si existe más de una evaluación en la época, conserva la última.
        eval_columns = [
            column for column in epoch_logs.columns
            if column.startswith("eval_")
        ]
        if eval_columns:
            eval_logs = epoch_logs[epoch_logs[eval_columns].notna().any(axis=1)]
            if not eval_logs.empty:
                last_eval = eval_logs.iloc[-1]
                for metric in eval_columns:
                    value = last_eval.get(metric)
                    if pd.notna(value) and isinstance(value, (int, float)):
                        row[metric] = float(value)

        row.update(priority_config)
        rows.append(row)

    epoch_metrics = pd.DataFrame(rows)
    if epoch_metrics.empty:
        raise RuntimeError("No fue posible construir las métricas por época.")

    preferred_order = [
        "run_id",
        "dataset_epoch",
        "train_loss",
        "eval_loss",
        "eval_accuracy",
        "eval_margin",
        "eval_rewards/chosen",
        "eval_rewards/rejected",
        "train_grad_norm_mean",
        "train_grad_norm_max",
        "learning_rate_end",
    ]
    ordered = [
        column for column in preferred_order if column in epoch_metrics.columns
    ]
    remaining = [
        column for column in epoch_metrics.columns if column not in ordered
    ]
    epoch_metrics = epoch_metrics[ordered + remaining]

    return epoch_metrics


def save_experiment_datasets(
    trainer: RewardTrainer,
    config: dict[str, Any],
    train_dataset_size: int,
    eval_dataset_size: int,
    run_id: str,
) -> Path:
    """
    Guarda un único CSV con una fila por época.

    El CSV integra las principales métricas de entrenamiento y evaluación con
    los hiperparámetros prioritarios. No se crean CSV por paso ni archivos
    Excel, para mantener un único dataset tabular por entrenamiento.
    """
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_metrics = build_epoch_metrics_dataset(
        log_history=trainer.state.log_history,
        config=config,
        train_dataset_size=train_dataset_size,
        eval_dataset_size=eval_dataset_size,
        run_id=run_id,
    )

    epoch_path = output_dir / config["logging"]["epoch_metrics_filename"]
    config_path = output_dir / config["logging"]["config_filename"]

    epoch_metrics.to_csv(epoch_path, index=False, encoding="utf-8-sig")

    config_to_save = {
        **config,
        "resolved": {
            "run_id": run_id,
            "train_dataset_size": train_dataset_size,
            "eval_dataset_size": eval_dataset_size,
            "world_size": int(torch.distributed.get_world_size()) if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ) else 1,
        },
    }
    config_path.write_text(
        json.dumps(config_to_save, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"CSV único de métricas por época: {epoch_path.resolve()}")
    print(f"Configuración ejecutada: {config_path.resolve()}")

    return epoch_path


def print_training_summary(
    config: dict[str, Any],
    train_dataset: Dataset,
    eval_dataset: Dataset,
    dtype: torch.dtype,
) -> None:
    """Muestra los hiperparámetros principales antes de entrenar."""
    world_size = int(torch.distributed.get_world_size()) if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ) else 1
    effective_batch = (
        config["training"]["per_device_train_batch_size"]
        * config["training"]["gradient_accumulation_steps"]
        * world_size
    )

    print("\n======= CONFIGURACIÓN REWARD MODEL =======")
    print(f"Dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "Memoria GPU: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    print(f"Precisión: {dtype}")
    print(f"Preferencias de entrenamiento: {len(train_dataset)}")
    print(f"Preferencias de evaluación: {len(eval_dataset)}")
    print(f"Épocas: {config['training']['num_train_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Batch efectivo: {effective_batch}")
    print(f"Longitud máxima: {config['training']['max_length']}")
    print(f"Center rewards: {config['training']['center_rewards_coefficient']}")
    print(f"LoRA r: {config['lora']['r']}")
    print(
        "Gradient checkpointing: "
        f"{config['performance']['gradient_checkpointing']}"
    )
    print("==========================================\n")


def main(config: dict[str, Any]) -> None:
    """Entrena, evalúa, fusiona y exporta el Reward Model."""
    seed = config["training"]["seed"]
    set_seed(seed)
    configure_performance(config)

    dtype, use_bf16, use_fp16 = select_dtype()
    tf32_enabled = (
        config["performance"]["allow_tf32"]
        and supports_tf32()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["paths"]["model"],
        use_fast=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        config["paths"]["model"],
        num_labels=1,
        torch_dtype=dtype,
        attn_implementation=config[
            "performance"
        ]["attention_implementation"],
        low_cpu_mem_usage=config["performance"]["low_cpu_mem_usage"],
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not config[
        "performance"
    ]["gradient_checkpointing"]

    train_dataset, eval_dataset = load_preference_dataset(
        csv_path=config["paths"]["preferences"],
        data_config=config["data"],
        seed=seed,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        target_modules=config["lora"]["target_modules"],
        modules_to_save=config["lora"]["modules_to_save"],
    )

    training_args = RewardConfig(
        output_dir=config["paths"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config[
            "training"
        ]["per_device_train_batch_size"],
        per_device_eval_batch_size=config[
            "training"
        ]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config[
            "training"
        ]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        max_length=config["training"]["max_length"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        max_grad_norm=config["training"]["max_grad_norm"],
        center_rewards_coefficient=config[
            "training"
        ]["center_rewards_coefficient"],
        gradient_checkpointing=config[
            "performance"
        ]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=tf32_enabled,
        optim=config["performance"]["optim"],
        torch_compile=config["performance"]["torch_compile"],
        dataloader_num_workers=config[
            "performance"
        ]["dataloader_num_workers"],
        dataloader_pin_memory=config[
            "performance"
        ]["dataloader_pin_memory"],
        dataloader_persistent_workers=config[
            "performance"
        ]["dataloader_persistent_workers"],
        eval_strategy=config["logging"]["eval_strategy"],
        logging_strategy=config["logging"]["logging_strategy"],
        save_strategy=config["logging"]["save_strategy"],
        save_total_limit=config["logging"]["save_total_limit"],
        load_best_model_at_end=config[
            "logging"
        ]["load_best_model_at_end"],
        metric_for_best_model=config[
            "logging"
        ]["metric_for_best_model"],
        greater_is_better=config["logging"]["greater_is_better"],
        report_to=config["logging"]["report_to"],
        seed=seed,
        data_seed=seed,
    )

    print_training_summary(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dtype=dtype,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.train()

    final_metrics = trainer.evaluate()
    trainer.save_metrics("eval", final_metrics)
    trainer.save_model(config["paths"]["output_dir"])
    tokenizer.save_pretrained(config["paths"]["output_dir"])

    save_experiment_datasets(
        trainer=trainer,
        config=config,
        train_dataset_size=len(train_dataset),
        eval_dataset_size=len(eval_dataset),
        run_id=run_id,
    )

    # PPO necesita un checkpoint autocontenido. Los adaptadores LoRA se
    # fusionan con el modelo base y se conserva el cabezal escalar.
    merged_dir = Path(config["paths"]["merged_output_dir"])
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_model = trainer.model.merge_and_unload()
    merged_model.config.pad_token_id = tokenizer.pad_token_id
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    with (merged_dir / "reward_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_metrics, file, indent=2, ensure_ascii=False)

    print(
        "Adaptador del Reward Model: "
        f"{Path(config['paths']['output_dir']).resolve()}"
    )
    print(f"Reward Model fusionado para PPO: {merged_dir.resolve()}")
    print(f"Métricas finales de evaluación: {final_metrics}")


if __name__ == "__main__":
    main(CONFIG)
