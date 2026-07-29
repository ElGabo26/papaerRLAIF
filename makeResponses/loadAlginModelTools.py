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


import tqdm


def generar_respuestas_batch(
    model,
    tokenizer,
    prompts,
    batch_size=120,
    max_new_tokens=200,
    num_responses=1
):
    """
    Genera respuestas para varios prompts simultáneamente.

    Parámetros
    ----------
    model:
        Modelo cargado.

    tokenizer:
        Tokenizador del modelo.

    prompts:
        Lista de prompts.

    batch_size:
        Cantidad de prompts enviados simultáneamente a la GPU.

    max_new_tokens:
        Máximo de tokens generados por respuesta.

    num_responses:
        Número de respuestas generadas por cada prompt.

    Retorna
    -------
    list:
        Lista de respuestas. Cuando num_responses > 1,
        retorna una lista de listas.
    """

    model.eval()

    # Para modelos causales se recomienda padding por la izquierda.
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    responses = []

    for inicio in tqdm.tqdm(
        range(0, len(prompts), batch_size),
        desc="Generando respuestas por lotes"
    ):
        batch_prompts = prompts[inicio:inicio + batch_size]

        # Aplicar el chat template a cada prompt.
        textos = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for prompt in batch_prompts
        ]

        inputs = tokenizer(
            textos,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Enviar tensores al dispositivo del modelo.
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=num_responses > 1,
                temperature=0.7 if num_responses > 1 else None,
                top_p=0.90 if num_responses > 1 else None,
                num_return_sequences=num_responses,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Eliminar los tokens correspondientes al prompt.
        generated_tokens = outputs[:, input_length:]

        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        decoded = [
            response.strip()
            for response in decoded
        ]

        if num_responses == 1:
            responses.extend(decoded)

        else:
            # Agrupar las respuestas pertenecientes a cada prompt.
            for indice in range(len(batch_prompts)):
                inicio_respuestas = indice * num_responses
                fin_respuestas = inicio_respuestas + num_responses

                responses.append(
                    decoded[inicio_respuestas:fin_respuestas]
                )

    return responses