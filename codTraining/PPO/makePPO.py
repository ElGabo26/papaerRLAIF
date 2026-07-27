import os

import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer


# ============================================================
# 1. Configuración
# ============================================================

TOKENIZER_ID = "EleutherAI/pythia-1b-deduped"

POLICY_MODEL_ID = (
    "cleanrl/"
    "EleutherAI_pythia-1b-deduped__sft__tldr"
)

REWARD_MODEL_ID = (
    "cleanrl/"
    "EleutherAI_pythia-1b-deduped__reward__tldr"
)

DATASET_ID = "trl-lib/tldr"
OUTPUT_DIR = "./pythia-1b-ppo-lora"

SEED = 42
MAX_PROMPT_LENGTH = 512

set_seed(SEED)


if not torch.cuda.is_available():
    raise RuntimeError(
        "Este ejemplo está configurado para una GPU CUDA."
    )


use_bf16 = torch.cuda.is_bf16_supported()

compute_dtype = (
    torch.bfloat16
    if use_bf16
    else torch.float16
)


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Precisión: {compute_dtype}")


# ============================================================
# 2. Tokenizador
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID,
    padding_side="left",
)

# Evita crear un token nuevo fuera del vocabulario.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 3. Cargar el dataset de prompts
# ============================================================

dataset = load_dataset(DATASET_ID)

# Para una demostración pequeña utilizamos un subconjunto.
train_dataset = dataset["train"].select(
    range(min(256, len(dataset["train"])))
)


def tokenize_prompt(example: dict) -> dict:
    """
    PPOTrainer espera una columna input_ids con el prompt
    tokenizado.
    """

    input_ids = tokenizer(
        example["prompt"],
        padding=False,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
        add_special_tokens=True,
    )["input_ids"]

    return {
        "input_ids": input_ids,
        "lengths": len(input_ids),
    }


# PPOTrainer únicamente necesita input_ids y lengths.
with PartialState().local_main_process_first():
    train_dataset = train_dataset.map(
        tokenize_prompt,
        remove_columns=train_dataset.column_names,
        num_proc=1,
    )


# Eliminar prompts vacíos o demasiado largos.
train_dataset = train_dataset.filter(
    lambda example: (
        0 < example["lengths"] <= MAX_PROMPT_LENGTH
    )
)


# El script oficial espera que el prompt no termine en EOS,
# porque PPO generará la continuación.
train_dataset = train_dataset.filter(
    lambda example: (
        example["input_ids"][-1]
        != tokenizer.eos_token_id
    )
)


print(train_dataset)
print("Primer prompt tokenizado:")
print(train_dataset[0]["input_ids"][:20])


# ============================================================
# 4. Argumentos de carga de modelos
# ============================================================

model_kwargs = {
    "dtype": compute_dtype,
    "low_cpu_mem_usage": True,
}


# ============================================================
# 5. Modelo de política
# ============================================================

policy_model = AutoModelForCausalLM.from_pretrained(
    POLICY_MODEL_ID,
    **model_kwargs,
)

policy_model.config.pad_token_id = tokenizer.pad_token_id
policy_model.config.use_cache = False


# ============================================================
# 6. Reward Model
# ============================================================

reward_model = (
    AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_ID,
        num_labels=1,
        **model_kwargs,
    )
)

reward_model.config.pad_token_id = tokenizer.pad_token_id


# El Reward Model no se actualiza durante PPO.
for parameter in reward_model.parameters():
    parameter.requires_grad = False

reward_model.eval()


# ============================================================
# 7. Value Model
# ============================================================

# El modelo de valor estima V(s), es decir, el retorno
# esperado para cada estado/token.
value_model = (
    AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_ID,
        num_labels=1,
        **model_kwargs,
    )
)

value_model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# 8. Configuración LoRA
# ============================================================

peft_config = LoraConfig(
    task_type="CAUSAL_LM",

    # Rango de la adaptación.
    r=16,

    # Escala LoRA.
    lora_alpha=32,

    # Regularización.
    lora_dropout=0.05,

    bias="none",

    # PEFT buscará las capas lineales del modelo.
    target_modules="all-linear",
)


# ============================================================
# 9. Configuración PPO
# ============================================================

ppo_config = PPOConfig(
    output_dir=OUTPUT_DIR,

    # PPO suele usar una tasa de aprendizaje pequeña.
    learning_rate=3e-6,

    # Configuración de lote para una GPU.
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    # Cantidad total de prompts/respuestas utilizados.
    # Para un experimento real debe ser mucho mayor.
    total_episodes=128,

    # Veces que PPO reutiliza cada rollout.
    num_ppo_epochs=2,

    # Número de divisiones internas del lote.
    num_mini_batches=1,

    # Máximo de tokens generados por respuesta.
    response_length=53,

    # Controla el tamaño de lote durante generación.
    local_rollout_forward_batch_size=1,

    # Generación estocástica.
    temperature=0.7,

    # Finalizar cuando se produzca EOS.
    stop_token="eos",

    # Penalizar respuestas que no produzcan EOS.
    missing_eos_penalty=1.0,

    # Penalización por alejarse del modelo de referencia.
    kl_coef=0.05,

    # Recorte de la actualización de política.
    cliprange=0.2,

    # Recorte de las actualizaciones de valor.
    cliprange_value=0.2,

    # Peso de la pérdida del modelo de valor.
    vf_coef=0.1,

    # Parámetros para Generalized Advantage Estimation.
    gamma=1.0,
    lam=0.95,

    # Reducción de memoria.
    gradient_checkpointing=True,

    # Precisión mixta.
    bf16=use_bf16,
    fp16=not use_bf16,

    # Registro.
    logging_steps=1,
    report_to="none",

    # Guardado.
    save_strategy="steps",
    save_steps=32,
    save_total_limit=2,

    # No utilizamos evaluación en este ejemplo.
    eval_strategy="no",

    seed=SEED,
)


# ============================================================
# 10. Crear PPOTrainer
# ============================================================

trainer = PPOTrainer(
    args=ppo_config,

    # Tokenizador.
    processing_class=tokenizer,

    # Política que será entrenada.
    model=policy_model,

    # Cuando se usa PEFT puede mantenerse como None.
    # TRL administra la política de referencia.
    ref_model=None,

    # Modelo que entrega la recompensa.
    reward_model=reward_model,

    # Modelo que estima el valor esperado.
    value_model=value_model,

    # Dataset de prompts.
    train_dataset=train_dataset,

    # Configuración LoRA.
    peft_config=peft_config,
)


# ============================================================
# 11. Mostrar parámetros entrenables
# ============================================================

if hasattr(trainer.model, "print_trainable_parameters"):
    trainer.model.print_trainable_parameters()


# ============================================================
# 12. Entrenamiento PPO
# ============================================================

train_result = trainer.train()

print("\nResultado del entrenamiento:")
print(train_result)


# ============================================================
# 13. Guardar adaptadores
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModelo PPO guardado en: {OUTPUT_DIR}")