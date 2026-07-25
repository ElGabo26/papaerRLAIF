import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tqdm.auto import tqdm


# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT_FILE = ""
OUTPUT_FILE = ""

PROMPT_COLUMN = ""
ID_COLUMN = ""

# Puede sustituirlo por otro modelo disponible en su proyecto.
MODEL = "gpt-5.6"

# Máximo de tokens generados por respuesta.
MAX_OUTPUT_TOKENS = 200

# Número máximo de intentos por prompt.
MAX_RETRIES = 6

# Espera inicial para el backoff exponencial.
INITIAL_WAIT_SECONDS = 2.0

# Instrucción general aplicada a todos los prompts.
SYSTEM_INSTRUCTIONS = (
    "Respond clearly, accurately, and only to the user's request."
)


# ============================================================
# VALIDACIONES
# ============================================================

def validate_environment() -> None:
    """
    Comprueba que la clave de OpenAI esté configurada.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "No se encontró la variable OPENAI_API_KEY.\n"
            "Linux/RunPod:\n"
            '  export OPENAI_API_KEY="su_clave"\n\n'
            "Windows PowerShell:\n"
            '  setx OPENAI_API_KEY "su_clave"'
        )


def load_prompts(file_path: Path) -> pd.DataFrame:
    """
    Lee y valida el archivo que contiene los prompts.
    """
    df = pd.read_csv(file_path)
    df[PROMPT_COLUMN] = df[PROMPT_COLUMN].astype(str).str.strip()
    df = df[df[PROMPT_COLUMN] != ""].copy()
    return df.reset_index(drop=True)


# ============================================================
# USO DE TOKENS
# ============================================================

def get_usage_value(
    usage: Any,
    attribute: str,
) -> int | None:
    """
    Obtiene de manera segura un valor del objeto usage.
    """
    if usage is None:
        return None

    value = getattr(usage, attribute, None)

    if value is None and isinstance(usage, dict):
        value = usage.get(attribute)

    return value


# ============================================================
# CONSULTA A OPENAI
# ============================================================

def request_openai(
    client: OpenAI,
    prompt: str,
) -> dict[str, Any]:
    """
    Envía un prompt al modelo y devuelve la respuesta junto
    con sus metadatos.

    Utiliza reintentos con espera exponencial ante errores
    temporales de conexión, límite de solicitudes o servidor.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_INSTRUCTIONS,
                input=prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

            input_tokens = get_usage_value(
                response.usage,
                "input_tokens",
            )

            output_tokens = get_usage_value(
                response.usage,
                "output_tokens",
            )

            total_tokens = get_usage_value(
                response.usage,
                "total_tokens",
            )

            return {
                "response": response.output_text,
                "status": "success",
                "error": None,
                "response_id": response.id,
                "model": MODEL,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "attempts": attempt,
            }

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as error:
            last_error = error

            if attempt == MAX_RETRIES:
                break

            # Backoff exponencial:
            # 2, 4, 8, 16, 32 segundos.
            wait_time = INITIAL_WAIT_SECONDS * (2 ** (attempt - 1))

            tqdm.write(
                f"Error temporal: {type(error).__name__}. "
                f"Reintento {attempt}/{MAX_RETRIES} "
                f"en {wait_time:.1f} segundos."
            )

            time.sleep(wait_time)

        except APIStatusError as error:
            # Algunos errores HTTP no deben reintentarse,
            # por ejemplo una API key inválida.
            return {
                "response": None,
                "status": "error",
                "error": (
                    f"{type(error).__name__}: "
                    f"HTTP {error.status_code} - {error}"
                ),
                "response_id": None,
                "model": MODEL,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "attempts": attempt,
            }

        except Exception as error:
            return {
                "response": None,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "response_id": None,
                "model": MODEL,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "attempts": attempt,
            }

    return {
        "response": None,
        "status": "error",
        "error": (
            f"{type(last_error).__name__}: {last_error}"
            if last_error
            else "Se agotaron los reintentos."
        ),
        "response_id": None,
        "model": MODEL,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "attempts": MAX_RETRIES,
    }


# ============================================================
# CONTROL DE RESULTADOS Y REANUDACIÓN
# ============================================================

def load_previous_results(
    output_file: Path,
) -> pd.DataFrame:
    """
    Lee los resultados existentes para reanudar una ejecución
    interrumpida.
    """
    if not output_file.exists():
        return pd.DataFrame()

    try:
        previous = pd.read_csv(output_file)

        if ID_COLUMN not in previous.columns:
            raise ValueError(
                f"El archivo {output_file} no contiene la columna "
                f"'{ID_COLUMN}'."
            )

        return previous

    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_results(
    results: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """
    Guarda todos los resultados acumulados.
    """
    result_df = pd.DataFrame(results)

    # Escritura temporal para reducir el riesgo de corrupción.
    temporary_file = output_file.with_suffix(".tmp.csv")

    result_df.to_csv(
        temporary_file,
        index=False,
        encoding="utf-8-sig",
    )

    temporary_file.replace(output_file)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def process_prompts() -> pd.DataFrame:
    """
    Procesa los prompts y almacena los resultados.
    """
    validate_environment()

    prompts_df = load_prompts(INPUT_FILE)
    previous_df = load_previous_results(OUTPUT_FILE)

    if previous_df.empty:
        results: list[dict[str, Any]] = []
        processed_ids: set[str] = set()
    else:
        results = previous_df.to_dict(orient="records")

        # Se convierten los IDs a string para evitar diferencias
        # entre valores como 1 y "1".
        processed_ids = set(
            previous_df[ID_COLUMN].astype(str)
        )

    pending_df = prompts_df[
        ~prompts_df[ID_COLUMN].astype(str).isin(processed_ids)
    ].copy()

    print("=" * 60)
    print("PROCESAMIENTO DE PROMPTS CON OPENAI")
    print("=" * 60)
    print(f"Modelo:              {MODEL}")
    print(f"Prompts encontrados: {len(prompts_df)}")
    print(f"Ya procesados:       {len(processed_ids)}")
    print(f"Pendientes:          {len(pending_df)}")
    print(f"Archivo de salida:   {OUTPUT_FILE.resolve()}")
    print("=" * 60)

    if pending_df.empty:
        print("Todos los prompts ya fueron procesados.")
        return previous_df

    # timeout: tiempo máximo de espera por solicitud.
    # max_retries=0: usamos nuestros propios reintentos.
    client = OpenAI(
        timeout=120.0,
        max_retries=0,
    )

    for _, row in tqdm(
        pending_df.iterrows(),
        total=len(pending_df),
        desc="Enviando prompts",
        unit="prompt",
    ):
        prompt_id = row[ID_COLUMN]
        prompt = row[PROMPT_COLUMN]

        started_at = time.perf_counter()

        api_result = request_openai(
            client=client,
            prompt=prompt,
        )

        elapsed_seconds = time.perf_counter() - started_at

        record = {
            ID_COLUMN: prompt_id,
            PROMPT_COLUMN: prompt,
            **api_result,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "processed_at": pd.Timestamp.now(
                tz="America/Guayaquil"
            ).isoformat(),
        }

        results.append(record)

        # Se guarda después de cada prompt. Así puede reanudarse
        # la ejecución sin perder resultados.
        save_results(
            results=results,
            output_file=OUTPUT_FILE,
        )

        if api_result["status"] == "error":
            tqdm.write(
                f"Error en prompt {prompt_id}: "
                f"{api_result['error']}"
            )

    final_df = pd.DataFrame(results)

    successful = int(
        (final_df["status"] == "success").sum()
    )

    errors = int(
        (final_df["status"] == "error").sum()
    )

    total_tokens = pd.to_numeric(
        final_df.get("total_tokens"),
        errors="coerce",
    ).sum()

    print("\n" + "=" * 60)
    print("PROCESAMIENTO FINALIZADO")
    print("=" * 60)
    print(f"Respuestas correctas: {successful}")
    print(f"Errores:               {errors}")
    print(f"Tokens utilizados:     {int(total_tokens)}")
    print(f"Resultado guardado en: {OUTPUT_FILE.resolve()}")
    print("=" * 60)

    return final_df


if __name__ == "__main__":
    process_prompts()