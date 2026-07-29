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
MODEL_PATH="/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_MODEL_PATH="/workspace/adaptedModels/PPO/finalPPOModels/DeepSeek-R1-Distill-Qwen-1.5B-PPO"
REWARD_MODEL_PATH="/workspace/adaptedModels/PPO/mergedRewardModels/DeepSeek-R1-Distill-Qwen-1.5B"

from typing import Any

CONFIG: dict[str, Any] = {
    "paths": {
        "sft_model": MODEL_PATH,
        "reward_model": REWARD_MODEL_PATH,
        "prompts": PREFERENCES_CSV,
        "output_dir": OUTPUT_MODEL_PATH,
    },

    "data": {
        # Mantiene una evaluación pequeña. Usa 0.0 solo si tu código admite
        # entrenar sin conjunto de evaluación.
        "test_size": 0.05,

        # Aumentar si la máquina dispone de suficientes núcleos de CPU.
        "num_proc": 8,

        # Reduce episodios redundantes.
        "remove_duplicates": True,
    },

    "training": {
        # Se calcula como prompts_train × epochs_over_dataset.
        "total_episodes": None,

        # Un único recorrido por el dataset.
        "epochs_over_dataset": 1,

        "seed": 42,
    },

    "batch": {
        # Incrementa el trabajo simultáneo de la GPU.
        # Con 67 % de VRAM inicial, 4 es un punto de partida razonable.
        "per_device_train_batch_size": 4,

        # Evita varios forward/backward antes de actualizar.
        "gradient_accumulation_steps": 1,

        # Parámetro especialmente importante para acelerar la generación PPO.
        # Reducir a 8 si aparece CUDA OOM.
        "local_rollout_forward_batch_size": 16,

        # Una sola división interna del batch.
        "num_mini_batches": 1,

        # Ajustar según los núcleos reales de la máquina.
        "dataloader_num_workers": 8,

        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
    },

    "ppo": {
        # Se incrementa ligeramente porque solo se ejecutará una época PPO.
        "learning_rate": 5e-6,

        # Una única actualización PPO por rollout.
        "num_ppo_epochs": 1,

        # Estos parámetros no reducen directamente el tiempo por paso,
        # pero se conservan en valores estables.
        "kl_coef": 0.05,
        "cliprange": 0.20,
        "cliprange_value": 0.20,
        "vf_coef": 0.10,
        "gamma": 1.0,
        "lam": 0.95,
        "max_grad_norm": 1.0,

        # Elimina pasos de calentamiento.
        "warmup_ratio": 0.0,

        # Evita el cálculo del scheduler coseno.
        "lr_scheduler_type": "constant",
    },

    "generation": {
        # Principal reducción del costo de generación.
        "response_length": 64,

        # Mantiene diversidad suficiente para PPO.
        "temperature": 0.70,

        # Permite terminar antes de los 64 tokens.
        "stop_token": "eos",

        "missing_eos_penalty": 1.0,
    },

    "baseline": {
        # Activa la evaluación del modelo SFT antes de aplicar PPO.
        # Esta fila permite comparar si PPO mejora o empeora la recompensa base.
        "enabled": True,

        # Usa evaluación para medir generalización inicial. Cambia a "train"
        # solo si deseas medir la base sobre los mismos prompts de entrenamiento.
        "dataset_split": "eval",

        # Limita el costo de la medición base. Usa None para evaluar todos los
        # prompts de la partición seleccionada.
        "max_samples": 128,

        # Batch usado solo para generar y puntuar la línea base.
        # Aumentarlo acelera la evaluación si la VRAM lo permite.
        "batch_size": 4,
    },

    "lora": {
        # Reduce parámetros entrenables y costo del backward.
        "r": 8,

        # Mantiene alpha/r = 2.
        "alpha": 16,

        # Dropout cero es ligeramente más rápido.
        "dropout": 0.0,

        "bias": "none",

        # Mucho más eficiente que entrenar todas las capas lineales.
        # Compatible con arquitecturas que usan q_proj y v_proj.
        "target_modules": [
            "q_proj",
            "v_proj",
        ],
    },

    "performance": {
        # Buen equilibrio entre compatibilidad y rendimiento.
        "attention_implementation": "sdpa",

        "allow_tf32": True,

        # Más rápido mientras exista suficiente VRAM.
        "gradient_checkpointing": False,

        "optim": "adamw_torch_fused",

        # La generación dinámica suele provocar recompilaciones.
        "torch_compile": False,

        "low_cpu_mem_usage": True,
    },

    "logging": {
        # Reduce escrituras y sincronizaciones CPU/GPU.
        "logging_steps": 20,

        # Desactiva evaluación durante el entrenamiento.
        # El código debe admitir "no".
        "eval_strategy": "no",

        # Evita crear checkpoints intermedios.
        "save_strategy": "no",

        # No se utiliza mientras save_strategy sea "no".
        "save_total_limit": 1,

        # Elimina la sobrecarga de TensorBoard.
        "report_to": "none",

        "epoch_metrics_filename": (
            f"ppo_metrics_by_epoch_{MODEL_PATH.split('/')[-1]}.csv"
        ),

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
    "config_baseline_enabled": ("baseline", "enabled"),
    "config_baseline_split": ("baseline", "dataset_split"),
    "config_baseline_max_samples": ("baseline", "max_samples"),
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


def compute_baseline_metrics(
    config: dict[str, Any],
    tokenizer: AutoTokenizer,
    policy: AutoModelForCausalLM,
    reward_model: AutoModelForSequenceClassification,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> dict[str, Any]:
    """
    Evalúa el modelo SFT antes de PPO usando el Reward Model.

    La métrica base principal es la recompensa promedio de las respuestas
    generadas por la política inicial. Esta fila queda en el mismo CSV que las
    épocas PPO para comparar el punto de partida contra el modelo alineado.
    """
    baseline_config = config["baseline"]
    split_name = baseline_config["dataset_split"]
    if split_name == "train":
        dataset = train_dataset
    elif split_name == "eval":
        dataset = eval_dataset
    else:
        raise ValueError("baseline.dataset_split debe ser 'train' o 'eval'.")

    if len(dataset) == 0:
        raise ValueError("La partición seleccionada para baseline está vacía.")

    max_samples = baseline_config["max_samples"]
    sample_count = len(dataset) if max_samples is None else min(
        int(max_samples),
        len(dataset),
    )
    dataset = dataset.select(range(sample_count))

    device = PartialState().device
    policy.to(device)
    reward_model.to(device)
    policy.eval()
    reward_model.eval()

    batch_size = int(baseline_config["batch_size"])
    max_new_tokens = int(config["generation"]["response_length"])
    temperature = float(config["generation"]["temperature"])

    reward_values: list[float] = []
    response_lengths: list[int] = []
    eos_count = 0

    old_use_cache = getattr(policy.config, "use_cache", None)
    policy.config.use_cache = True

    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            batch = dataset[start : start + batch_size]
            prompts = [{"input_ids": ids} for ids in batch["input_ids"]]
            prompt_tensors = tokenizer.pad(
                prompts,
                padding=True,
                return_tensors="pt",
            ).to(device)

            generated = policy.generate(
                **prompt_tensors,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            prompt_width = prompt_tensors["input_ids"].shape[1]
            attention_mask = torch.ones_like(generated, dtype=torch.long)
            attention_mask[:, :prompt_width] = prompt_tensors["attention_mask"]
            rewards = reward_model(
                input_ids=generated,
                attention_mask=attention_mask,
            ).logits.squeeze(-1)

            reward_values.extend(
                rewards.detach().float().cpu().tolist()
            )

            for row_index in range(generated.shape[0]):
                response = generated[row_index, prompt_width:]
                if (
                    tokenizer.eos_token_id is not None
                    and (response == tokenizer.eos_token_id).any().item()
                ):
                    eos_count += 1
                response = response[
                    response != tokenizer.pad_token_id
                ]
                response_lengths.append(int(response.numel()))

    if old_use_cache is not None:
        policy.config.use_cache = old_use_cache

    rewards_tensor = torch.tensor(reward_values, dtype=torch.float32)
    lengths_tensor = torch.tensor(response_lengths, dtype=torch.float32)

    return {
        "baseline_split": split_name,
        "baseline_num_samples": sample_count,
        "baseline_reward_mean": float(rewards_tensor.mean().item()),
        "baseline_reward_std": float(
            rewards_tensor.std(unbiased=False).item()
        ),
        "baseline_reward_min": float(rewards_tensor.min().item()),
        "baseline_reward_max": float(rewards_tensor.max().item()),
        "baseline_response_length_mean": float(lengths_tensor.mean().item()),
        "baseline_response_length_max": int(lengths_tensor.max().item()),
        "baseline_eos_rate": eos_count / sample_count,
    }


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
        self.baseline_metrics: dict[str, Any] | None = None

    def set_baseline_metrics(self, metrics: dict[str, Any]) -> None:
        """Registra las métricas del modelo SFT antes de entrenar PPO."""
        self.baseline_metrics = metrics

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

        if self.baseline_metrics:
            baseline_row = {
                "run_id": self.run_id,
                "phase": "baseline",
                "dataset_epoch": 0,
                "global_step_start": 0,
                "global_step_end": 0,
                "logged_updates": 0,
                "episodes_start_estimated": 0,
                "episodes_end_estimated": 0,
                **self.baseline_metrics,
                **priority_config,
            }
            epoch_metrics.insert(2, "phase", "ppo")
            epoch_metrics = pd.concat(
                [pd.DataFrame([baseline_row]), epoch_metrics],
                ignore_index=True,
                sort=False,
            )

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

    if config["baseline"]["enabled"]:
        baseline_metrics = compute_baseline_metrics(
            config=config,
            tokenizer=tokenizer,
            policy=policy,
            reward_model=reward_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        metrics_callback.set_baseline_metrics(baseline_metrics)
        print("\n========== MÉTRICAS BASE ANTES DE PPO ==========")
        print(f"Partición: {baseline_metrics['baseline_split']}")
        print(f"Muestras: {baseline_metrics['baseline_num_samples']}")
        print(
            "Recompensa media: "
            f"{baseline_metrics['baseline_reward_mean']:.4f}"
        )
        print(
            "Longitud media de respuesta: "
            f"{baseline_metrics['baseline_response_length_mean']:.2f} tokens"
        )
        print(f"EOS rate: {baseline_metrics['baseline_eos_rate']:.4f}")
        print("================================================\n")

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
