import os
import json
import torch
import pandas as pd

from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import DPOConfig, DPOTrainer


# ============================================================
# 1. Configuración general
# ============================================================

MODEL_ID = input("Ruta del modelo base: ").strip()
OUTPUT_DIR = input("Ruta del modelo de salida: ").strip()
PREFERENCES = input("Ruta del archivo CSV de preferencias: ").strip()

SEED = 42
TEST_SIZE = 0.25

set_seed(SEED)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU no disponible. Se utilizará CPU.")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

print(f"BF16 disponible: {use_bf16}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Dataset de preferencias
# ============================================================

df = pd.read_csv(PREFERENCES, index_col=0)

required_columns = {"prompt", "chosen", "rejected"}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"El CSV debe contener las columnas {sorted(required_columns)}. "
        f"Faltan: {sorted(missing_columns)}"
    )

# Eliminar filas con valores nulos en las columnas necesarias.
df = df.dropna(subset=["prompt", "chosen", "rejected"]).reset_index(drop=True)

# Eliminar duplicados exactos para evitar ponderar artificialmente ejemplos repetidos.
df = df.drop_duplicates(
    subset=["prompt", "chosen", "rejected"]
).reset_index(drop=True)

dataset = Dataset.from_pandas(
    df,
    preserve_index=False,
)

dataset = dataset.train_test_split(
    test_size=TEST_SIZE,
    seed=SEED,
)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"\nRegistros totales: {len(df)}")
print(f"Registros de entrenamiento: {len(train_dataset)}")
print(f"Registros de evaluación: {len(eval_dataset)}")


# ============================================================
# 3. Tokenizador
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")


# ============================================================
# 4. Modelo base
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
)

# Necesario para gradient checkpointing.
model.config.use_cache = False


# ============================================================
# 5. Configuración LoRA
# ============================================================

peft_config = LoraConfig(
    task_type="CAUSAL_LM",

    # Rango de las matrices LoRA.
    r=8,

    # Factor de escalamiento de LoRA.
    lora_alpha=16,

    # Regularización.
    lora_dropout=0.0,

    # No entrenar bias del modelo base.
    bias="none",

    # Aplicar LoRA a todas las capas lineales elegibles.
    target_modules=[

            "q_proj",

            "v_proj",

        ],
)


# ============================================================
# 6. Configuración DPO
# ============================================================

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # Reference policy
    # --------------------------------------------------------
    precompute_ref_log_probs=True,
    precompute_ref_batch_size=8,

    # --------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------
    num_train_epochs=5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=4,

    learning_rate=1e-5,

    # --------------------------------------------------------
    # DPO
    # --------------------------------------------------------
    beta=0.1,
    loss_type="sigmoid",

    # Longitud total máxima: prompt + respuesta.
    max_length=512,

    # --------------------------------------------------------
    # Memoria y precisión
    # --------------------------------------------------------
    gradient_checkpointing=True,

    bf16=use_bf16,
    fp16=torch.cuda.is_available() and not use_bf16,

    # --------------------------------------------------------
    # Evaluación y checkpoints
    # --------------------------------------------------------
    eval_strategy="epoch",
    save_strategy="epoch",

    # Restaurar automáticamente el mejor checkpoint.
    load_best_model_at_end=True,

    # Seleccionar el mejor checkpoint usando eval_loss.
    metric_for_best_model="eval_loss",

    # En eval_loss, menor es mejor.
    greater_is_better=False,

    # Mantener un número limitado de checkpoints.
    save_total_limit=2,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    logging_strategy="steps",
    logging_steps=1,
    report_to="none",

    # --------------------------------------------------------
    # Reproducibilidad
    # --------------------------------------------------------
    seed=SEED,
)


# ============================================================
# 7. Crear DPOTrainer
# ============================================================

trainer = DPOTrainer(
    model=model,

    # Con PEFT, TRL puede usar la política inicial como
    # referencia sin mantener necesariamente una segunda
    # copia completa del modelo.
    ref_model=None,

    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    processing_class=tokenizer,

    # DPOTrainer añadirá los adaptadores LoRA.
    peft_config=peft_config,
)


# ============================================================
# 8. Mostrar parámetros entrenables
# ============================================================

print("\n========================================")
print("PARÁMETROS ENTRENABLES")
print("========================================")

trainer.model.print_trainable_parameters()


# ============================================================
# 9. Entrenamiento
# ============================================================

print("\n========================================")
print("INICIO DEL ENTRENAMIENTO DPO")
print("========================================")

train_result = trainer.train()

print("\nMétricas globales de entrenamiento:")
print(train_result.metrics)


# ============================================================
# 10. Analizar historial y obtener mejor época
# ============================================================

log_history = trainer.state.log_history

best_epoch = None
best_eval_loss = float("inf")

evaluation_history = []

for log in log_history:
    if "eval_loss" in log:
        epoch = log.get("epoch")
        eval_loss = float(log["eval_loss"])

        evaluation_history.append(
            {
                "epoch": epoch,
                "eval_loss": eval_loss,
            }
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch


print("\n========================================")
print("MEJOR MODELO DPO")
print("========================================")

print(f"Mejor época: {best_epoch}")
print(f"Mejor eval_loss: {best_eval_loss}")
print(f"Mejor checkpoint: {trainer.state.best_model_checkpoint}")


# ============================================================
# 11. Evaluación final
# ============================================================
#
# Como load_best_model_at_end=True, trainer.model contiene
# en este punto los pesos del mejor checkpoint según eval_loss.
# ============================================================

final_eval_metrics = trainer.evaluate()

print("\n========================================")
print("MÉTRICAS DEL MEJOR MODELO")
print("========================================")

for key, value in final_eval_metrics.items():
    print(f"{key}: {value}")


# ============================================================
# 12. Guardar mejor modelo y tokenizer
# ============================================================

best_model_dir = os.path.join(
    OUTPUT_DIR,
    "best_model"
)

os.makedirs(
    best_model_dir,
    exist_ok=True,
)

# Guarda los adaptadores LoRA del mejor checkpoint.
trainer.save_model(best_model_dir)

# Guarda el tokenizer.
tokenizer.save_pretrained(best_model_dir)


# ============================================================
# 13. Guardar historial de entrenamiento
# ============================================================

log_history_path = os.path.join(
    OUTPUT_DIR,
    "training_log_history.json",
)

with open(
    log_history_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        log_history,
        f,
        indent=4,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# 14. Guardar métricas por época
# ============================================================

evaluation_history_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_history.csv",
)

pd.DataFrame(
    evaluation_history
).to_csv(
    evaluation_history_path,
    index=False,
)


# ============================================================
# 15. Guardar resumen del experimento
# ============================================================

experiment_summary = {
    "model_id": MODEL_ID,
    "preferences_file": PREFERENCES,

    "seed": SEED,

    "dataset": {
        "total_examples": len(df),
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "test_size": TEST_SIZE,
    },

    "dpo": {
        "num_train_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "beta": training_args.beta,
        "loss_type": training_args.loss_type,
        "max_length": training_args.max_length,

        "per_device_train_batch_size":
            training_args.per_device_train_batch_size,

        "per_device_eval_batch_size":
            training_args.per_device_eval_batch_size,

        "gradient_accumulation_steps":
            training_args.gradient_accumulation_steps,

        "effective_batch_size_single_gpu":
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps,
    },

    "lora": {
        "r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "lora_dropout": peft_config.lora_dropout,
        "bias": peft_config.bias,
        "target_modules": str(peft_config.target_modules),
    },

    "precision": {
        "bf16": bool(use_bf16),
        "fp16": bool(
            torch.cuda.is_available()
            and not use_bf16
        ),
    },

    "best_model": {
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss,
        "best_checkpoint":
            trainer.state.best_model_checkpoint,
        "saved_path": best_model_dir,
    },

    "final_evaluation": final_eval_metrics,
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "experiment_summary.json",
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        experiment_summary,
        f,
        indent=4,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# 16. Resultado final
# ============================================================

print("\n========================================")
print("ENTRENAMIENTO FINALIZADO")
print("========================================")

print(f"Mejor época: {best_epoch}")
print(f"Mejor eval_loss: {best_eval_loss}")
print(f"Modelo guardado en: {best_model_dir}")
print(f"Historial JSON: {log_history_path}")
print(f"Métricas por época: {evaluation_history_path}")
print(f"Resumen experimental: {summary_path}")