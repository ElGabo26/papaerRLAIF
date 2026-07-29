import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def cargar_modelo(ruta_modelo: str):
    """
    Carga un modelo completo formado por:
    - modelo base;
    - adaptador LoRA entrenado con DPO o PPO;
    - tokenizador.

    La ruta debe contener adapter_config.json y
    adapter_model.safetensors.

    Retorna:
        model: modelo base con el adaptador LoRA cargado.
        tokenizer: tokenizador correspondiente.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        ruta_modelo,
        local_files_only=True
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        ruta_modelo,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    return model, tokenizer





def generar_respuesta(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200
) -> str:
    """
    Recibe un modelo, su tokenizador y un prompt.
    Retorna únicamente la respuesta generada.
    """

    mensajes = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        mensajes,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    # Envía los tensores al mismo dispositivo del modelo.
    inputs = {
        clave: valor.to(model.device)
        for clave, valor in inputs.items()
    }

    with torch.inference_mode():
        salida = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Excluye los tokens correspondientes al prompt.
    tokens_respuesta = salida[
        0,
        inputs["input_ids"].shape[1]:
    ]

    respuesta = tokenizer.decode(
        tokens_respuesta,
        skip_special_tokens=True
    )

    return respuesta.strip()