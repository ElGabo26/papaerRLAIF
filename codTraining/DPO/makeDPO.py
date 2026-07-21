import os
import torch
import pandas as  pd
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

MODEL_ID = input("ruta  del  modelo: ")
OUTPUT_DIR = input("ruta  del  modelo de  salida: ")
SEED = 42
PREFERENCES=input("ruta preferencias ")

set_seed(SEED)

print(f"GPU: {torch.cuda.get_device_name(0)}")

use_bf16 = torch.cuda.is_bf16_supported()

compute_dtype = (
    torch.bfloat16
    if use_bf16
    else torch.float16
)

# ============================================================
# 2. Dataset de preferencias
# ============================================================

# Crear un Dataset de Hugging Face.
df=pd.read_csv(PREFERENCES, index_col=0)
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(
    test_size=0.25,
    seed=SEED,
)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Registros de entrenamiento: {len(train_dataset)}")
print(f"Registros de evaluación: {len(eval_dataset)}")


# ============================================================
# 3. Tokenizador
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

# DPOTrainer requiere padding izquierdo.
tokenizer.padding_side = "left"

# Algunos modelos no tienen pad_token definido.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4. Modelo base con Transformers
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype='auto',
)

# Debe desactivarse durante gradient checkpointing.
model.config.use_cache = False


# ============================================================
# 5. Configuración LoRA con PEFT
# ============================================================

peft_config = LoraConfig(
    task_type="CAUSAL_LM",

    # Rango de las matrices LoRA.
    r=16,

    # Escalamiento aplicado a la actualización LoRA.
    lora_alpha=32,

    # Regularización.
    lora_dropout=0.05,

    # No entrenar los sesgos del modelo base.
    bias="none",

    # Aplica LoRA a las capas lineales del Transformer.
    # PEFT excluye normalmente la capa de salida.
    target_modules="all-linear",
)


# ============================================================
# 6. Configuración DPO con TRL
# ============================================================

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,

    # Configuración del entrenamiento.
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,

    # Para adaptadores LoRA suele emplearse una tasa mayor
    # que en full fine-tuning.
    learning_rate=1e-5,

    # Parámetro principal de DPO.
    beta=0.1,

    # Objetivo DPO original.
    loss_type="sigmoid",

    # Longitud máxima: prompt + respuesta.
    max_length=512,

    # Reducción de memoria.
    gradient_checkpointing=True,

    # Precisión de entrenamiento.
    bf16=use_bf16,
    fp16=not use_bf16,

    # Evaluación y guardado.
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    # Registro.
    logging_strategy="steps",
    logging_steps=1,
    report_to="none",

    # Reproducibilidad.
    seed=SEED,
)


# ============================================================
# 7. Crear DPOTrainer
# ============================================================

trainer = DPOTrainer(
    # Modelo Transformers que será adaptado.
    model=model,

    # No se pasa ref_model.
    # TRL utilizará como referencia la política inicial.
    ref_model=None,

    # Argumentos DPO.
    args=training_args,

    # Datos de preferencia.
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    # Tokenizador.
    processing_class=tokenizer,

    # DPOTrainer añadirá los adaptadores LoRA al modelo.
    peft_config=peft_config,
)


# Mostrar cuántos parámetros serán entrenados.
trainer.model.print_trainable_parameters()


# ============================================================
# 8. Entrenamiento
# ============================================================

train_result = trainer.train()

log_history = trainer.state.log_history

for registro in log_history:
    print(registro)

print("\nMétricas de entrenamiento:")
print(train_result.metrics)


# ============================================================
# 9. Evaluación
# ============================================================

evaluation_metrics = trainer.evaluate()

print("\nMétricas de evaluación:")
print(evaluation_metrics)


# ============================================================
# 10. Guardar adaptadores LoRA
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nAdaptadores guardados en: {OUTPUT_DIR}")