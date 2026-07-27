from __future__ import annotations

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
from dotenv import load_dotenv

# Carga las variables del archivo .env al entorno
load_dotenv()

# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT_FILE = Path(r"C:\Users\Gabo\Desktop\PAPPER Rlaif\makeResponses\promptBases\finalPromptBases\elementary_math_prompts_1200.csv")
OUTPUT_FILE = Path(r"C:\Users\Gabo\Desktop\PAPPER Rlaif\makeResponses\responses\math_elementary_responses_deepSeek.csv")

PROMPT_COLUMN = "prompt"
ID_COLUMN = "id_prompt"

# Modelos disponibles:
# - deepseek-v4-flash: menor costo y mayor velocidad.
# - deepseek-v4-pro: mayor capacidad.
MODEL = "deepseek-v4-flash"

MAX_PROMPTS = 2000
MAX_TOKENS = 200

# Temperatura de generación.
TEMPERATURE = 0.3

# True: activa el modo de razonamiento.
# False: utiliza el modo sin razonamiento.
USE_THINKING = False

# Cantidad máxima de intentos por prompt.
MAX_RETRIES = 6

# Tiempo inicial del backoff exponencial.
INITIAL_WAIT_SECONDS = 2.0

SYSTEM_INSTRUCTIONS = (
    "Respond clearly, accurately, and only to the user's request."
)


# ============================================================
# VALIDACIÓN
# ============================================================

def validate_environment() -> None:
    """
    Verifica que la clave de DeepSeek esté configurada.
    """
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise EnvironmentError(
            "No se encontró la variable DEEPSEEK_API_KEY.\n\n"
            "Linux o RunPod:\n"
            'export DEEPSEEK_API_KEY="tu_clave"\n\n'
            "Windows PowerShell:\n"
            'setx DEEPSEEK_API_KEY "tu_clave"'
        )


def load_prompts(file_path: Path) -> pd.DataFrame:
    """
    Lee y valida el archivo CSV con los prompts.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {file_path.resolve()}"
        )

    df = pd.read_csv(file_path)

    if PROMPT_COLUMN not in df.columns:
        raise ValueError(
            f"El archivo debe contener la columna "
            f"'{PROMPT_COLUMN}'.\n"
            f"Columnas encontradas: {df.columns.tolist()}"
        )

    # Elimina valores nulos.
    df = df.dropna(subset=[PROMPT_COLUMN]).copy()

    # Convierte cada prompt a texto.
    df[PROMPT_COLUMN] = (
        df[PROMPT_COLUMN]
        .astype(str)
        .str.strip()
    )

    # Elimina prompts vacíos.
    df = df[df[PROMPT_COLUMN] != ""].copy()

    # Crea una columna ID cuando no existe.
    if ID_COLUMN not in df.columns:
        df.insert(
            0,
            ID_COLUMN,
            range(1, len(df) + 1),
        )

    # Verifica que no existan IDs duplicados.
    if df[ID_COLUMN].duplicated().any():
        duplicated_ids = df.loc[
            df[ID_COLUMN].duplicated(),
            ID_COLUMN,
        ].tolist()

        raise ValueError(
            f"Existen IDs duplicados: {duplicated_ids[:10]}"
        )

    return df.head(MAX_PROMPTS).reset_index(drop=True)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def get_usage_value(
    usage: Any,
    attribute: str,
) -> int | None:
    """
    Extrae de manera segura un valor del objeto usage.
    """
    if usage is None:
        return None

    value = getattr(usage, attribute, None)

    if value is None and isinstance(usage, dict):
        value = usage.get(attribute)

    return value


def empty_result(
    error: str,
    attempts: int,
    total_elapsed_seconds: float,
    api_time_seconds: float,
    retry_wait_seconds: float,
) -> dict[str, Any]:
    """
    Construye un resultado estándar para solicitudes fallidas.
    """
    return {
        "response": None,
        "reasoning_content": None,
        "status": "error",
        "error": error,
        "response_id": None,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "thinking_enabled": USE_THINKING,
        "max_tokens": MAX_TOKENS,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "elapsed_seconds": round(
            total_elapsed_seconds,
            4,
        ),
        "api_time_seconds": round(
            api_time_seconds,
            4,
        ),
        "retry_wait_seconds": round(
            retry_wait_seconds,
            4,
        ),
        "attempts": attempts,
    }


# ============================================================
# SOLICITUD A DEEPSEEK
# ============================================================

def request_deepseek(
    client: OpenAI,
    prompt: str,
) -> dict[str, Any]:
    """
    Envía un prompt a DeepSeek y devuelve la respuesta,
    el consumo de tokens, la configuración y los tiempos.

    Implementa reintentos con backoff exponencial.
    """
    last_error: Exception | None = None

    total_start_time = time.perf_counter()

    api_time_seconds = 0.0
    retry_wait_seconds = 0.0

    for attempt in range(1, MAX_RETRIES + 1):
        attempt_start_time = time.perf_counter()

        try:
            # Configuración específica del modo thinking.
            extra_body = {
                "thinking": {
                    "type": (
                        "enabled"
                        if USE_THINKING
                        else "disabled"
                    )
                }
            }

            request_parameters: dict[str, Any] = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_INSTRUCTIONS,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "stream": False,
                "extra_body": extra_body,
            }

            # reasoning_effort solamente resulta relevante
            # cuando el modo thinking está activado.
            if USE_THINKING:
                request_parameters["reasoning_effort"] = "high"

            response = client.chat.completions.create(
                **request_parameters
            )

            attempt_elapsed = (
                time.perf_counter()
                - attempt_start_time
            )

            api_time_seconds += attempt_elapsed

            total_elapsed_seconds = (
                time.perf_counter()
                - total_start_time
            )

            message = response.choices[0].message

            response_text = message.content

            reasoning_content = getattr(
                message,
                "reasoning_content",
                None,
            )

            prompt_tokens = get_usage_value(
                response.usage,
                "prompt_tokens",
            )

            completion_tokens = get_usage_value(
                response.usage,
                "completion_tokens",
            )

            total_tokens = get_usage_value(
                response.usage,
                "total_tokens",
            )

            return {
                "response": response_text,
                "reasoning_content": reasoning_content,
                "status": "success",
                "error": None,
                "response_id": response.id,
                "model": response.model,
                "temperature": TEMPERATURE,
                "thinking_enabled": USE_THINKING,
                "max_tokens": MAX_TOKENS,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "elapsed_seconds": round(
                    total_elapsed_seconds,
                    4,
                ),
                "api_time_seconds": round(
                    api_time_seconds,
                    4,
                ),
                "retry_wait_seconds": round(
                    retry_wait_seconds,
                    4,
                ),
                "attempts": attempt,
            }

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as error:
            attempt_elapsed = (
                time.perf_counter()
                - attempt_start_time
            )

            api_time_seconds += attempt_elapsed
            last_error = error

            if attempt == MAX_RETRIES:
                break

            # Backoff exponencial:
            # 2, 4, 8, 16 y 32 segundos.
            wait_time = (
                INITIAL_WAIT_SECONDS
                * (2 ** (attempt - 1))
            )

            tqdm.write(
                f"Error temporal: {type(error).__name__}. "
                f"Reintento {attempt}/{MAX_RETRIES} "
                f"en {wait_time:.1f} segundos."
            )

            time.sleep(wait_time)

            retry_wait_seconds += wait_time

        except APIStatusError as error:
            attempt_elapsed = (
                time.perf_counter()
                - attempt_start_time
            )

            api_time_seconds += attempt_elapsed

            total_elapsed_seconds = (
                time.perf_counter()
                - total_start_time
            )

            return empty_result(
                error=(
                    f"{type(error).__name__}: "
                    f"HTTP {error.status_code} - {error}"
                ),
                attempts=attempt,
                total_elapsed_seconds=total_elapsed_seconds,
                api_time_seconds=api_time_seconds,
                retry_wait_seconds=retry_wait_seconds,
            )

        except Exception as error:
            attempt_elapsed = (
                time.perf_counter()
                - attempt_start_time
            )

            api_time_seconds += attempt_elapsed

            total_elapsed_seconds = (
                time.perf_counter()
                - total_start_time
            )

            return empty_result(
                error=f"{type(error).__name__}: {error}",
                attempts=attempt,
                total_elapsed_seconds=total_elapsed_seconds,
                api_time_seconds=api_time_seconds,
                retry_wait_seconds=retry_wait_seconds,
            )

    total_elapsed_seconds = (
        time.perf_counter()
        - total_start_time
    )

    return empty_result(
        error=(
            f"{type(last_error).__name__}: {last_error}"
            if last_error
            else "Se agotaron todos los reintentos."
        ),
        attempts=MAX_RETRIES,
        total_elapsed_seconds=total_elapsed_seconds,
        api_time_seconds=api_time_seconds,
        retry_wait_seconds=retry_wait_seconds,
    )


# ============================================================
# GUARDADO Y REANUDACIÓN
# ============================================================

def load_previous_results(
    output_file: Path,
) -> pd.DataFrame:
    """
    Lee resultados anteriores para reanudar el proceso.
    """
    if not output_file.exists():
        return pd.DataFrame()

    try:
        previous_df = pd.read_csv(output_file)

        if ID_COLUMN not in previous_df.columns:
            raise ValueError(
                f"El archivo {output_file} no contiene "
                f"la columna '{ID_COLUMN}'."
            )

        return previous_df

    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_results(
    results: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """
    Guarda progresivamente los resultados en un CSV.
    """
    result_df = pd.DataFrame(results)

    temporary_file = output_file.with_suffix(
        ".tmp.csv"
    )

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
    Procesa hasta 1000 prompts con DeepSeek.
    """
    validate_environment()

    prompts_df = load_prompts(INPUT_FILE)

    previous_df = load_previous_results(
        OUTPUT_FILE
    )

    if previous_df.empty:
        results: list[dict[str, Any]] = []
        processed_ids: set[str] = set()
    else:
        results = previous_df.to_dict(
            orient="records"
        )

        processed_ids = set(
            previous_df[ID_COLUMN].astype(str)
        )

    pending_df = prompts_df[
        ~prompts_df[ID_COLUMN]
        .astype(str)
        .isin(processed_ids)
    ].copy()

    print("=" * 65)
    print("PROCESAMIENTO DE PROMPTS CON DEEPSEEK")
    print("=" * 65)
    print(f"Modelo:                  {MODEL}")
    print(f"Modo thinking:           {USE_THINKING}")
    print(f"Temperatura:             {TEMPERATURE}")
    print(f"Prompts encontrados:     {len(prompts_df)}")
    print(f"Prompts ya procesados:   {len(processed_ids)}")
    print(f"Prompts pendientes:      {len(pending_df)}")
    print(f"Archivo de salida:       {OUTPUT_FILE.resolve()}")
    print("=" * 65)

    if pending_df.empty:
        print("Todos los prompts ya fueron procesados.")
        return previous_df

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=180.0,

        # Los reintentos se controlan manualmente.
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

        api_result = request_deepseek(
            client=client,
            prompt=prompt,
        )

        record = {
            ID_COLUMN: prompt_id,
            PROMPT_COLUMN: prompt,
            **api_result,
            "processed_at": pd.Timestamp.now(
                tz="America/Guayaquil"
            ).isoformat(),
        }

        results.append(record)

        # Guarda después de cada solicitud para poder
        # reanudar el proceso si se interrumpe.
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

    total_elapsed = pd.to_numeric(
        final_df.get("elapsed_seconds"),
        errors="coerce",
    ).sum()

    print("\n" + "=" * 65)
    print("PROCESAMIENTO FINALIZADO")
    print("=" * 65)
    print(f"Respuestas correctas:    {successful}")
    print(f"Errores:                 {errors}")
    print(f"Tokens totales:          {int(total_tokens)}")
    print(f"Tiempo acumulado:        {total_elapsed:.2f} segundos")
    print(f"Archivo generado:        {OUTPUT_FILE.resolve()}")
    print("=" * 65)

    return final_df


if __name__ == "__main__":
    process_prompts()