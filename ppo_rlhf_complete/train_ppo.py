from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer


# ============================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ============================================================
#
# Los parámetros están agrupados por función. Los parámetros marcados como
# PRIORITARIOS son los primeros que conviene variar durante la experimentación.

PREFERENCES_CSV="/workspace/papaerRLAIF/codTraining/calify/preferencesQwen.csv"
MODEL_PATH="/workspace/models/Qwen2.5-1.5B-Instruct"
OUTPUT_MODEL_PATH="/workspace/adaptedModels/PPO/finalPPOModels/Qwen2.5-1.5B-Instruct-PPO"
REWARD_MODEL_PATH="/workspace/adaptedModels/PPO/mergedRewardModels/Qwen2.5-1.5B-Instruct"

CONFIG: dict[str, Any] = {
    "paths": {
        # Modelo Instruct o SFT que se utilizará como política inicial de PPO.
        "sft_model": MODEL_PATH,

        # Reward Model fusionado. Debe producir una única puntuación escalar.
        "reward_model": REWARD_MODEL_PATH,

        # CSV de entrada. PPO utiliza la columna "prompt".
        "prompts": PREFERENCES_CSV,

        # Directorio del modelo PPO, checkpoints, métricas y configuración.
        "output_dir": OUTPUT_MODEL_PATH,
    },
    "data": {
        # Fracción de prompts reservada para evaluación.
        "test_size": 0.20,

        # Procesos de CPU utilizados para tokenizar los prompts.
        "num_proc": 4,

        # Elimina prompts duplicados para evitar ponderarlos varias veces.
        "remove_duplicates": True,
    },
    "training": {
        # Total exacto de episodios. Si es None, se calcula como:
        # prompts_train × epochs_over_dataset.
        "total_episodes": None,

        # PRIORITARIO. Recorridos completos sobre los prompts de entrenamiento.
        # No debe confundirse con las épocas internas de PPO.
        "epochs_over_dataset": 3,

        # Semilla para que la división de datos y el entrenamiento sean
        # reproducibles.
        "seed": 42,
    },
    "batch": {
        # Prompts procesados simultáneamente por GPU.
        # Si aparece CUDA OOM, reducir de 2 a 1.
        "per_device_train_batch_size": 4,

        # Pasos acumulados antes de actualizar los parámetros.
        "gradient_accumulation_steps": 4,

        # Respuestas procesadas simultáneamente durante los rollouts.
        # Incrementarlo mejora la utilización de GPU si existe VRAM disponible.
        "local_rollout_forward_batch_size": 4,

        # Divisiones internas del batch de PPO.
        "num_mini_batches": 1,

        # Trabajadores de CPU empleados por el DataLoader.
        "dataloader_num_workers": 6,

        # Acelera la transferencia CPU → GPU.
        "dataloader_pin_memory": True,

        # Conserva los trabajadores entre iteraciones.
        "dataloader_persistent_workers": True,
    },
    "ppo": {
        # PRIORITARIO. Magnitud de las actualizaciones de la política.
        "learning_rate": 3e-6,

        # PRIORITARIO. Actualizaciones PPO realizadas sobre cada rollout.
        # Un valor alto reutiliza más cada rollout, pero puede desestabilizar PPO.
        "num_ppo_epochs": 4,

        # PRIORITARIO. Penalización por alejarse de la política SFT.
        "kl_coef": 0.05,

        # PRIORITARIO. Limita cambios grandes en la política.
        "cliprange": 0.20,

        # Limita cambios grandes en el Value Model.
        "cliprange_value": 0.20,

        # Peso de la pérdida del Value Model.
        "vf_coef": 0.10,

        # Factor de descuento de las recompensas.
        "gamma": 1.0,

        # Parámetro de GAE para el balance entre sesgo y varianza.
        "lam": 0.95,

        # Norma máxima de los gradientes para evitar explosiones.
        "max_grad_norm": 1.0,

        # Fracción inicial usada para incrementar gradualmente el learning rate.
        "warmup_ratio": 0.03,

        # Evolución del learning rate durante el entrenamiento.
        "lr_scheduler_type": "cosine",
    },
    "generation": {
        # PRIORITARIO. Máximo de tokens generados en cada respuesta.
        "response_length": 128,

        # PRIORITARIO. Diversidad de las respuestas usadas como rollouts.
        "temperature": 0.70,

        # Detiene la generación cuando aparece el token EOS.
        "stop_token": "eos",

        # Penalización si una respuesta no termina con EOS.
        "missing_eos_penalty": 1.0,
    },
    "lora": {
        # PRIORITARIO. Rango y capacidad de los adaptadores LoRA.
        "r": 16,

        # Factor de escalamiento de las actualizaciones LoRA.
        "alpha": 32,

        # Regularización aplicada a los adaptadores.
        "dropout": 0.05,

        # No se entrenan los términos bias del modelo base.
        "bias": "none",

        # Aplica LoRA a las capas lineales del Transformer.
        "target_modules": "all-linear",
    },
    "performance": {
        # Implementación optimizada de atención incluida en PyTorch.
        "attention_implementation": "sdpa",

        # Acelera operaciones matriciales en GPUs NVIDIA modernas.
        "allow_tf32": True,

        # False es más rápido. Cambiar a True únicamente si falta VRAM.
        "gradient_checkpointing": False,

        # Implementación fusionada de AdamW.
        "optim": "adamw_torch_fused",

        # PPO usa generación dinámica; se desactiva para evitar recompilaciones.
        "torch_compile": False,

        # Reduce el consumo máximo de RAM durante la carga de modelos.
        "low_cpu_mem_usage": True,
    },
    "logging": {
        # Se registra cada actualización para poder agregar métricas por época.
        # Cambiarlo a 5 o 10 reduce ligeramente la sobrecarga, pero ofrece una
        # estimación menos detallada.
        "logging_steps": 1,

        # Evalúa y guarda al finalizar cada recorrido del dataset.
        "eval_strategy": "epoch",
        "save_strategy": "epoch",

        # Máximo de checkpoints retenidos.
        "save_total_limit": 2,

        # TensorBoard conserva las métricas interactivas.
        "report_to": "tensorboard",

        # Nombre del CSV resumido: una fila por época del dataset.
        "epoch_metrics_filename": f"ppo_metrics_by_epoch_{MODEL_PATH.split('/')[-1]}.csv",

        # Copia serializada del diccionario de configuración ejecutado.
        "config_filename": "ppo_experiment_config.json",
    },
}


PRIORITY_CONFIG_COLUMNS = {
    "config_sft_model": ("paths", "sft_model"),
    "config_reward_model": ("paths", "reward_model"),
    "config_learning_rate": ("ppo", "learning_rate"),
    "config_epochs_over_dataset": ("training", "epochs_over_dataset"),
    "config_num_ppo_epochs": ("ppo", "num_ppo_epochs"),
    "config_kl_coef": ("ppo", "kl_coef"),
    "config_cliprange": ("ppo", "cliprange"),
    "config_cliprange_value": ("ppo", "cliprange_value"),
    "config_vf_coef": ("ppo", "vf_coef"),
    "config_gamma": ("ppo", "gamma"),
    "config_lam": ("ppo", "lam"),
    "config_train_batch_size": ("batch", "per_device_train_batch_size"),
    "config_gradient_accumulation": ("batch", "gradient_accumulation_steps"),
    "config_num_mini_batches": ("batch", "num_mini_batches"),
    "config_response_length": ("generation", "response_length"),
    "config_temperature": ("generation", "temperature"),
    "config_missing_eos_penalty": ("generation", "missing_eos_penalty"),
    "config_lora_r": ("lora", "r"),
    "config_lora_alpha": ("lora", "alpha"),
    "config_lora_dropout": ("lora", "dropout"),
    "config_seed": ("training", "seed"),
}


def nested_value(config: dict[str, Any], path: tuple[str, str]) -> Any:
    """Obtiene un valor de un diccionario con dos niveles."""
    return config[path[0]][path[1]]


def select_dtype() -> tuple[torch.dtype, bool, bool]:
    """Selecciona BF16, FP16 o FP32 según las capacidades del equipo."""
    if not torch.cuda.is_available():
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def configure_gpu(config: dict[str, Any]) -> None:
    """Activa kernels rápidos que no modifican el tamaño del dataset."""
    allow_tf32 = config["performance"]["allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def load_prompt_dataset(
    csv_path: str,
    tokenizer: AutoTokenizer,
    data_config: dict[str, Any],
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Carga, limpia, divide y tokeniza los prompts."""
    frame = pd.read_csv(csv_path)
    if "prompt" not in frame.columns:
        raise ValueError("El CSV debe contener una columna llamada 'prompt'.")

    frame = frame[["prompt"]].dropna().copy()
    frame["prompt"] = frame["prompt"].astype(str).str.strip()
    frame = frame[frame["prompt"] != ""]

    if data_config["remove_duplicates"]:
        frame = frame.drop_duplicates(subset=["prompt"])

    frame = frame.reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("Se necesitan al menos dos prompts válidos.")

    dataset = Dataset.from_pandas(frame, preserve_index=False)
    split = dataset.train_test_split(
        test_size=data_config["test_size"],
        seed=seed,
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch["prompt"]
        ]
        tokenized = tokenizer(
            formatted_prompts,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        return {"input_ids": tokenized["input_ids"]}

    def prepare(partition: Dataset) -> Dataset:
        return partition.map(
            tokenize_batch,
            batched=True,
            num_proc=data_config["num_proc"],
            remove_columns=partition.column_names,
            desc="Tokenizando prompts",
        )

    with PartialState().local_main_process_first():
        train_dataset = prepare(split["train"])
        eval_dataset = prepare(split["test"])

    return train_dataset, eval_dataset


class PPOEpochMetricsCallback(TrainerCallback):
    """
    Captura las métricas de PPO y crea un único dataset por época.

    TRL controla PPO mediante ``total_episodes`` y no mediante las épocas
    clásicas de Trainer. Por ello, cada log se asigna a una época del dataset
    según su progreso:

        época = ceil(global_step / max_steps × epochs_over_dataset)

    De esta forma, ``num_ppo_epochs`` permanece correctamente identificado como
    el número de actualizaciones internas realizadas sobre cada rollout.
    """

    def __init__(
        self,
        config: dict[str, Any],
        train_dataset_size: int,
        eval_dataset_size: int,
        total_episodes: int,
    ) -> None:
        self.config = config
        self.train_dataset_size = train_dataset_size
        self.eval_dataset_size = eval_dataset_size
        self.total_episodes = total_episodes
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs: list[dict[str, Any]] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Conserva únicamente valores escalares serializables."""
        if not state.is_world_process_zero or not logs:
            return

        row: dict[str, Any] = {
            "run_id": self.run_id,
            "global_step": int(state.global_step),
        }
        for key, value in logs.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.detach().float().cpu().item()
            if isinstance(value, (str, bool, int, float)) or value is None:
                row[key] = value
        self.logs.append(row)

    def _priority_configuration(self) -> dict[str, Any]:
        """Construye las columnas de configuración repetidas en cada época."""
        values = {
            name: nested_value(self.config, path)
            for name, path in PRIORITY_CONFIG_COLUMNS.items()
        }
        world_size = PartialState().num_processes
        values["config_effective_batch_size"] = (
            self.config["batch"]["per_device_train_batch_size"]
            * self.config["batch"]["gradient_accumulation_steps"]
            * world_size
        )
        values["config_total_episodes"] = self.total_episodes
        values["train_dataset_size"] = self.train_dataset_size
        values["eval_dataset_size"] = self.eval_dataset_size
        return values

    @staticmethod
    def _is_training_metric(column: str) -> bool:
        """Excluye metadatos finales que no representan el comportamiento PPO."""
        excluded = {
            "run_id",
            "global_step",
            "epoch",
            "dataset_epoch",
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        }
        return column not in excluded

    def build_datasets(
        self,
        state: TrainerState,
    ) -> pd.DataFrame:
        """Devuelve las métricas de PPO agregadas por época del dataset."""
        raw = pd.DataFrame(self.logs)
        if raw.empty:
            raise RuntimeError(
                "TRL no generó eventos de métricas. Verifica logging_steps y "
                "que el entrenamiento haya completado al menos una actualización."
            )

        # Un mismo global_step puede producir varios eventos separados, por
        # ejemplo uno de entrenamiento y otro de evaluación. Se combinan
        # tomando el último valor no nulo de cada columna, sin descartar
        # métricas registradas previamente en ese mismo paso.
        def last_non_null(values: pd.Series) -> Any:
            non_null = values.dropna()
            return non_null.iloc[-1] if not non_null.empty else None

        raw = (
            raw.sort_values("global_step")
            .groupby("global_step", as_index=False, sort=True)
            .agg(last_non_null)
        )

        max_steps = int(state.max_steps or raw["global_step"].max())
        if max_steps <= 0:
            max_steps = int(raw["global_step"].max())
        if max_steps <= 0:
            raise RuntimeError("No fue posible determinar el número de pasos PPO.")

        dataset_epochs = int(self.config["training"]["epochs_over_dataset"])
        raw["dataset_epoch"] = (
            ((raw["global_step"].clip(lower=1) - 1) * dataset_epochs)
            // max_steps
            + 1
        ).clip(upper=dataset_epochs)

        numeric_metrics = [
            column
            for column in raw.select_dtypes(include="number").columns
            if self._is_training_metric(column)
        ]

        rows: list[dict[str, Any]] = []
        priority_config = self._priority_configuration()

        for epoch_number in range(1, dataset_epochs + 1):
            epoch_logs = raw[raw["dataset_epoch"] == epoch_number]
            if epoch_logs.empty:
                continue

            epoch_row: dict[str, Any] = {
                "run_id": self.run_id,
                "dataset_epoch": epoch_number,
                "global_step_start": int(epoch_logs["global_step"].min()),
                "global_step_end": int(epoch_logs["global_step"].max()),
                "logged_updates": len(epoch_logs),
                "episodes_start_estimated": int(
                    math.floor(
                        (epoch_number - 1)
                        * self.total_episodes
                        / dataset_epochs
                    )
                    + 1
                ),
                "episodes_end_estimated": int(
                    math.floor(
                        epoch_number
                        * self.total_episodes
                        / dataset_epochs
                    )
                ),
            }

            # Las métricas se promedian dentro de cada recorrido del dataset.
            for metric in numeric_metrics:
                values = pd.to_numeric(epoch_logs[metric], errors="coerce")
                if values.notna().any():
                    epoch_row[metric] = float(values.mean())

            # Learning rate y episode representan estado final, no un promedio.
            for last_value_metric in ("learning_rate", "lr", "episode"):
                if last_value_metric in epoch_logs.columns:
                    values = epoch_logs[last_value_metric].dropna()
                    if not values.empty:
                        epoch_row[f"{last_value_metric}_end"] = values.iloc[-1]

            epoch_row.update(priority_config)
            rows.append(epoch_row)

        epoch_metrics = pd.DataFrame(rows)
        if epoch_metrics.empty:
            raise RuntimeError("No fue posible agregar las métricas por época.")

        return epoch_metrics

    def save(self, state: TrainerState) -> Path:
        """
        Guarda un único CSV con una fila por época del dataset.

        Cada fila combina las métricas PPO agregadas y los hiperparámetros
        prioritarios. Los eventos internos por paso se mantienen solamente en
        memoria mientras se construye este resumen.
        """
        if not state.is_world_process_zero:
            return Path()

        output_dir = Path(self.config["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        epoch_metrics = self.build_datasets(state)

        epoch_path = output_dir / self.config["logging"]["epoch_metrics_filename"]
        config_path = output_dir / self.config["logging"]["config_filename"]

        epoch_metrics.to_csv(epoch_path, index=False, encoding="utf-8-sig")

        config_to_save = {
            **self.config,
            "resolved": {
                "run_id": self.run_id,
                "train_dataset_size": self.train_dataset_size,
                "eval_dataset_size": self.eval_dataset_size,
                "total_episodes": self.total_episodes,
                "world_size": PartialState().num_processes,
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
    total_episodes: int,
    dtype: torch.dtype,
) -> None:
    """Muestra los valores principales antes de iniciar PPO."""
    world_size = PartialState().num_processes
    effective_batch_size = (
        config["batch"]["per_device_train_batch_size"]
        * config["batch"]["gradient_accumulation_steps"]
        * world_size
    )

    print("\n========== CONFIGURACIÓN PPO ==========")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "Memoria GPU: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
    print(f"Precisión: {dtype}")
    print(f"Prompts de entrenamiento: {len(train_dataset)}")
    print(f"Episodios totales: {total_episodes}")
    print(
        "Recorridos del dataset: "
        f"{config['training']['epochs_over_dataset']}"
    )
    print(f"Épocas PPO por rollout: {config['ppo']['num_ppo_epochs']}")
    print(f"Batch efectivo: {effective_batch_size}")
    print(f"Learning rate: {config['ppo']['learning_rate']}")
    print(f"Coeficiente KL: {config['ppo']['kl_coef']}")
    print(f"Clip range: {config['ppo']['cliprange']}")
    print(f"LoRA r: {config['lora']['r']}")
    print("=======================================\n")


def main(config: dict[str, Any]) -> None:
    """Carga los componentes, ejecuta PPO y exporta las métricas."""
    seed = config["training"]["seed"]
    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("PPO requiere GPU. PyTorch no detectó CUDA.")

    configure_gpu(config)
    dtype, use_bf16, use_fp16 = select_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        config["paths"]["sft_model"],
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": config[
            "performance"
        ]["attention_implementation"],
        "low_cpu_mem_usage": config["performance"]["low_cpu_mem_usage"],
    }

    policy = AutoModelForCausalLM.from_pretrained(
        config["paths"]["sft_model"],
        **model_kwargs,
    )
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.use_cache = not config[
        "performance"
    ]["gradient_checkpointing"]

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config["paths"]["reward_model"],
        num_labels=1,
        **model_kwargs,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        config["paths"]["reward_model"],
        num_labels=1,
        **model_kwargs,
    )

    reward_model.config.pad_token_id = tokenizer.pad_token_id
    value_model.config.pad_token_id = tokenizer.pad_token_id
    reward_model.config.use_cache = False
    value_model.config.use_cache = False

    # El Reward Model solo evalúa; el Value Model sí se actualiza durante PPO.
    reward_model.requires_grad_(False)
    reward_model.eval()

    if reward_model.config.vocab_size != policy.config.vocab_size:
        raise ValueError(
            "La política y el Reward Model tienen vocabularios diferentes. "
            "Deben partir del mismo modelo base y tokenizer."
        )

    train_dataset, eval_dataset = load_prompt_dataset(
        csv_path=config["paths"]["prompts"],
        tokenizer=tokenizer,
        data_config=config["data"],
        seed=seed,
    )

    total_episodes = config["training"]["total_episodes"]
    if total_episodes is None:
        total_episodes = (
            len(train_dataset)
            * config["training"]["epochs_over_dataset"]
        )

    policy_peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        target_modules=config["lora"]["target_modules"],
    )

    ppo_args = PPOConfig(
        output_dir=config["paths"]["output_dir"],
        per_device_train_batch_size=config[
            "batch"
        ]["per_device_train_batch_size"],
        gradient_accumulation_steps=config[
            "batch"
        ]["gradient_accumulation_steps"],
        num_mini_batches=config["batch"]["num_mini_batches"],
        local_rollout_forward_batch_size=config[
            "batch"
        ]["local_rollout_forward_batch_size"],
        learning_rate=config["ppo"]["learning_rate"],
        total_episodes=total_episodes,
        num_ppo_epochs=config["ppo"]["num_ppo_epochs"],
        kl_coef=config["ppo"]["kl_coef"],
        cliprange=config["ppo"]["cliprange"],
        cliprange_value=config["ppo"]["cliprange_value"],
        vf_coef=config["ppo"]["vf_coef"],
        gamma=config["ppo"]["gamma"],
        lam=config["ppo"]["lam"],
        max_grad_norm=config["ppo"]["max_grad_norm"],
        warmup_ratio=config["ppo"]["warmup_ratio"],
        lr_scheduler_type=config["ppo"]["lr_scheduler_type"],
        response_length=config["generation"]["response_length"],
        temperature=config["generation"]["temperature"],
        stop_token=config["generation"]["stop_token"],
        missing_eos_penalty=config[
            "generation"
        ]["missing_eos_penalty"],
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=config["performance"]["allow_tf32"],
        gradient_checkpointing=config[
            "performance"
        ]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=config["performance"]["optim"],
        torch_compile=config["performance"]["torch_compile"],
        dataloader_num_workers=config[
            "batch"
        ]["dataloader_num_workers"],
        dataloader_pin_memory=config["batch"]["dataloader_pin_memory"],
        dataloader_persistent_workers=config[
            "batch"
        ]["dataloader_persistent_workers"],
        logging_strategy="steps",
        logging_steps=config["logging"]["logging_steps"],
        eval_strategy=config["logging"]["eval_strategy"],
        save_strategy=config["logging"]["save_strategy"],
        save_total_limit=config["logging"]["save_total_limit"],
        report_to=config["logging"]["report_to"],
        seed=seed,
        data_seed=seed,
        sft_model_path=config["paths"]["sft_model"],
        reward_model_path=config["paths"]["reward_model"],
    )

    print_training_summary(
        config=config,
        train_dataset=train_dataset,
        total_episodes=total_episodes,
        dtype=dtype,
    )

    metrics_callback = PPOEpochMetricsCallback(
        config=config,
        train_dataset_size=len(train_dataset),
        eval_dataset_size=len(eval_dataset),
        total_episodes=total_episodes,
    )

    trainer = PPOTrainer(
        args=ppo_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=policy_peft_config,
    )
    trainer.add_callback(metrics_callback)

    trainer.train()
    trainer.save_model(config["paths"]["output_dir"])
    tokenizer.save_pretrained(config["paths"]["output_dir"])
    metrics_callback.save(trainer.state)

    output_path = Path(config["paths"]["output_dir"]).resolve()
    print(f"\nModelo PPO guardado en: {output_path}")


if __name__ == "__main__":
    main(CONFIG)
