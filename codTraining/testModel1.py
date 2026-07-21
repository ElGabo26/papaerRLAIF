import gc
from typing import Optional

import torch
from peft import AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Configuración
# ============================================================

MAX_NEW_TOKENS = 200

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    DTYPE = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
else:
    # FP16 normalmente no es adecuado para inferencia en CPU.
    DTYPE = torch.float32


def get_loading_arguments() -> dict:
    """
    Argumentos comunes para cargar modelos.
    """

    arguments = {
        "torch_dtype": DTYPE,
        "low_cpu_mem_usage": True,
    }

    if CUDA_AVAILABLE:
        arguments["device_map"] = "auto"

    return arguments


# ============================================================
# Tokenizador
# ============================================================

def load_tokenizer(
    model_path: str,
    fallback_path: Optional[str] = None,
):
    """
    Carga el tokenizador.

    Para un adaptador PEFT intenta primero cargarlo desde la
    carpeta del adaptador. Si allí no existe, utiliza el
    tokenizador del modelo base.
    """

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )

    except (OSError, ValueError) as error:

        if fallback_path is None:
            raise RuntimeError(
                f"No fue posible cargar el tokenizador desde: "
                f"{model_path}"
            ) from error

        print(
            "El adaptador no contiene el tokenizador. "
            f"Se utilizará el tokenizador del modelo base:\n"
            f"{fallback_path}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            fallback_path,
            use_fast=True,
        )

    # Algunos modelos causales no tienen token de padding.
    if tokenizer.pad_token_id is None:

        if tokenizer.eos_token_id is None:
            raise ValueError(
                "El tokenizador no tiene pad_token ni eos_token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ============================================================
# Carga de modelos
# ============================================================

def load_base_model(model_path: str):
    """
    Carga un modelo completo.

    Puede ser:
    - El modelo original o puro.
    - Un modelo DPO con pesos completos.
    - Un modelo DPO cuyo adaptador fue fusionado.
    """

    print("\nCargando modelo completo...")
    print(f"Ruta: {model_path}")
    print(f"Tipo de dato: {DTYPE}")

    tokenizer = load_tokenizer(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **get_loading_arguments(),
    )

    model.eval()

    return model, tokenizer


def load_dpo_adapter(adapter_path: str):
    """
    Carga un adaptador DPO entrenado mediante PEFT/LoRA.
    """

    print("\nCargando adaptador DPO/PEFT...")
    print(f"Ruta del adaptador: {adapter_path}")
    print(f"Tipo de dato: {DTYPE}")

    try:
        peft_config = PeftConfig.from_pretrained(adapter_path)

    except (OSError, ValueError) as error:
        raise RuntimeError(
            "La ruta indicada no parece contener un adaptador PEFT.\n"
            "Comprueba que exista el archivo adapter_config.json."
        ) from error

    base_model_path = peft_config.base_model_name_or_path

    if not base_model_path:
        raise ValueError(
            "No se encontró base_model_name_or_path dentro de "
            "adapter_config.json."
        )

    print(f"Modelo base detectado: {base_model_path}")

    tokenizer = load_tokenizer(
        model_path=adapter_path,
        fallback_path=base_model_path,
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        is_trainable=False,
        **get_loading_arguments(),
    )

    model.eval()

    return model, tokenizer, base_model_path


# ============================================================
# Preparación del prompt
# ============================================================

def prepare_inputs(tokenizer, prompt: str):
    """
    Aplica la plantilla conversacional cuando está disponible.
    Si el modelo no dispone de chat_template, tokeniza el prompt
    directamente.
    """

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    if tokenizer.chat_template:

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    else:
        print(
            "Advertencia: el tokenizador no contiene chat_template. "
            "Se tokenizará el texto directamente."
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )

    return inputs


def get_input_device(model) -> torch.device:
    """
    Obtiene el dispositivo donde se encuentran los embeddings.
    Es más seguro cuando se utiliza device_map='auto'.
    """

    embedding_layer = model.get_input_embeddings()
    device = embedding_layer.weight.device

    if device.type == "meta":
        return next(
            parameter.device
            for parameter in model.parameters()
            if parameter.device.type != "meta"
        )

    return device


# ============================================================
# Generación
# ============================================================

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Genera una respuesta determinista.
    """

    inputs = prepare_inputs(tokenizer, prompt)

    input_device = get_input_device(model)

    inputs = {
        name: tensor.to(input_device)
        for name, tensor in inputs.items()
    }

    generation_arguments = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generation_arguments,
        )

    prompt_length = inputs["input_ids"].shape[-1]

    generated_tokens = output_ids[
        0,
        prompt_length:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# Liberación de memoria
# ============================================================

def unload_model(model) -> None:
    """
    Libera la memoria utilizada por el modelo.
    """

    del model

    gc.collect()

    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================
# Modos de ejecución
# ============================================================

def test_base_model() -> None:
    """
    Prueba únicamente el modelo puro o completo.
    """

    model_path = input(
        "Ruta del modelo puro/completo: "
    ).strip()

    prompt = input(
        "Escribe una pregunta: "
    ).strip()

    model, tokenizer = load_base_model(model_path)

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
    )

    print("\n" + "=" * 60)
    print("RESPUESTA DEL MODELO PURO/COMPLETO")
    print("=" * 60)
    print(response)

    unload_model(model)


def test_dpo_model() -> None:
    """
    Prueba únicamente el adaptador DPO.
    """

    adapter_path = input(
        "Ruta del adaptador DPO: "
    ).strip()

    prompt = input(
        "Escribe una pregunta: "
    ).strip()

    model, tokenizer, _ = load_dpo_adapter(adapter_path)

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
    )

    print("\n" + "=" * 60)
    print("RESPUESTA DEL MODELO DPO")
    print("=" * 60)
    print(response)

    unload_model(model)


def compare_models() -> None:
    """
    Compara el modelo original con el adaptador DPO usando
    exactamente el mismo prompt.

    Los modelos se cargan secuencialmente para evitar mantener
    ambos simultáneamente en la GPU.
    """

    adapter_path = input(
        "Ruta del adaptador DPO: "
    ).strip()

    peft_config = PeftConfig.from_pretrained(adapter_path)

    detected_base_path = peft_config.base_model_name_or_path

    print(
        "\nModelo base registrado en el adaptador:\n"
        f"{detected_base_path}"
    )

    custom_base_path = input(
        "\nRuta alternativa del modelo base "
        "[Enter para utilizar la detectada]: "
    ).strip()

    base_model_path = (
        custom_base_path
        if custom_base_path
        else detected_base_path
    )

    prompt = input(
        "\nEscribe una pregunta: "
    ).strip()

    # --------------------------------------------------------
    # Modelo puro
    # --------------------------------------------------------

    base_model, base_tokenizer = load_base_model(
        base_model_path
    )

    base_response = generate_response(
        model=base_model,
        tokenizer=base_tokenizer,
        prompt=prompt,
    )

    unload_model(base_model)
    del base_tokenizer

    # --------------------------------------------------------
    # Modelo DPO
    # --------------------------------------------------------

    dpo_model, dpo_tokenizer, _ = load_dpo_adapter(
        adapter_path
    )

    dpo_response = generate_response(
        model=dpo_model,
        tokenizer=dpo_tokenizer,
        prompt=prompt,
    )

    unload_model(dpo_model)
    del dpo_tokenizer

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROMPT")
    print("=" * 70)
    print(prompt)

    print("\n" + "=" * 70)
    print("MODELO PURO")
    print("=" * 70)
    print(base_response)

    print("\n" + "=" * 70)
    print("MODELO MODIFICADO CON DPO")
    print("=" * 70)
    print(dpo_response)


# ============================================================
# Programa principal
# ============================================================

def main() -> None:

    print("=" * 60)
    print("EVALUACIÓN DE MODELOS")
    print("=" * 60)

    print("1. Probar modelo puro o completo")
    print("2. Probar adaptador DPO/LoRA")
    print("3. Comparar modelo puro contra modelo DPO")

    option = input(
        "\nSelecciona una opción [1, 2 o 3]: "
    ).strip()

    try:

        if option == "1":
            test_base_model()

        elif option == "2":
            test_dpo_model()

        elif option == "3":
            compare_models()

        else:
            print(
                "Opción no válida. Debes seleccionar 1, 2 o 3."
            )

    except KeyboardInterrupt:
        print("\nEjecución cancelada por el usuario.")

    except torch.cuda.OutOfMemoryError:
        print(
            "\nNo existe suficiente memoria GPU para cargar el modelo."
        )

        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()

    except Exception as error:
        print("\nSe produjo un error:")
        print(type(error).__name__)
        print(error)


if __name__ == "__main__":
    main()