from typing import Any
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy  as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import  TensorDataset, Subset
from tqdm.auto import tqdm


def  dowloadModel(modelroot:str, localRoot:str):
    nameModel=modelroot.split("/")[-1]
    localRoot=localRoot+"/"+nameModel
    ruta_modelo = snapshot_download(
    repo_id=modelroot,
    local_dir=localRoot)
    print("Modelo descargado en:", ruta_modelo)
    return ruta_modelo



def get_vector_cls(
    last_hidden_state: torch.Tensor,
) -> torch.Tensor:
    """
    Extrae el vector contextual correspondiente al primer token
    de cada secuencia.

    Parámetros
    ----------
    last_hidden_state:
        Tensor con forma:

        [batch_size, sequence_length, hidden_size]

    Retorna
    -------
    Tensor con forma:

        [batch_size, hidden_size]
    """

    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state debe tener tres dimensiones: "
            "[batch_size, sequence_length, hidden_size]."
        )

    if last_hidden_state.shape[1] == 0:
        raise ValueError(
            "La secuencia no contiene tokens."
        )

    # El índice 0 corresponde al primer token.
    vector_cls = last_hidden_state[:, 0, :]

    return vector_cls


def realizar_mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calcula el promedio de los vectores de los tokens válidos.

    Parámetros
    ----------
    last_hidden_state:
        Tensor con forma:

        [batch_size, sequence_length, hidden_size]

    attention_mask:
        Tensor con forma:

        [batch_size, sequence_length]

        Contiene:
            1 para tokens válidos.
            0 para tokens de padding.

    Retorna
    -------
    Tensor con forma:

        [batch_size, hidden_size]
    """

    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state debe tener forma "
            "[batch_size, sequence_length, hidden_size]."
        )

    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask debe tener forma "
            "[batch_size, sequence_length]."
        )

    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError(
            "Las dimensiones batch_size y sequence_length "
            "de last_hidden_state y attention_mask deben coincidir."
        )

    # [batch_size, sequence_length]
    #                ↓
    # [batch_size, sequence_length, 1]
    expanded_mask = attention_mask.unsqueeze(-1)

    # Convertir la máscara al mismo tipo numérico
    # de los vectores del modelo.
    expanded_mask = expanded_mask.to(
        dtype=last_hidden_state.dtype
    )

    # Los vectores de padding se multiplican por cero.
    masked_embeddings = (
        last_hidden_state * expanded_mask
    )

    # Sumar los vectores de los tokens válidos.
    sum_embeddings = masked_embeddings.sum(dim=1)

    # Contar los tokens válidos de cada secuencia.
    valid_token_count = expanded_mask.sum(dim=1)

    # Evitar una división entre cero.
    valid_token_count = valid_token_count.clamp(
        min=1.0
    )

    mean_embeddings = (
        sum_embeddings / valid_token_count
    )

    return mean_embeddings

def realizar_max_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Obtiene el valor máximo de cada dimensión entre todos
    los tokens válidos de la secuencia.

    Parámetros
    ----------
    last_hidden_state:
        Tensor con forma:

        [batch_size, sequence_length, hidden_size]

    attention_mask:
        Tensor con forma:

        [batch_size, sequence_length]

    Retorna
    -------
    Tensor con forma:

        [batch_size, hidden_size]
    """

    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state debe tener forma "
            "[batch_size, sequence_length, hidden_size]."
        )

    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask debe tener forma "
            "[batch_size, sequence_length]."
        )

    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError(
            "Las dimensiones batch_size y sequence_length "
            "de last_hidden_state y attention_mask deben coincidir."
        )

    # Comprobar que cada secuencia tenga al menos un token válido.
    if torch.any(attention_mask.sum(dim=1) == 0):
        raise ValueError(
            "Todas las secuencias deben contener al menos "
            "un token válido."
        )

    # [batch_size, sequence_length, 1]
    expanded_mask = attention_mask.unsqueeze(-1).bool()

    # Valor negativo extremadamente pequeño para el tipo
    # numérico del tensor.
    minimum_value = torch.finfo(
        last_hidden_state.dtype
    ).min

    # Reemplazar los tokens de padding por un valor muy negativo,
    # de modo que nunca sean seleccionados como máximos.
    masked_embeddings = last_hidden_state.masked_fill(
        ~expanded_mask,
        minimum_value,
    )

    # Obtener el máximo a través de la dimensión de tokens.
    max_embeddings = masked_embeddings.max(
        dim=1
    ).values

    return max_embeddings


@torch.inference_mode()
def obtener_salida_modelo(
    texto: str | list[str],
    tokenizer: Any,
    modelo: nn.Module,
    device: torch.device,
    max_length: int = 128,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """
    Tokeniza uno o varios textos y obtiene la salida del encoder.

    Parámetros
    ----------
    texto:
        Una cadena o una lista de cadenas.

    tokenizer:
        Tokenizador asociado exactamente al modelo utilizado.

    modelo:
        Modelo BERT, DeBERTa, RoBERTa u otro encoder compatible.

    device:
        Dispositivo en el que se ejecutará el modelo.

    max_length:
        Número máximo de tokens permitidos por secuencia.

    Retorna
    -------
    outputs:
        Salida completa del modelo.

    tokens:
        Diccionario con input_ids, attention_mask y otros
        tensores que requiera el modelo.
    """

    if isinstance(texto, str):
        textos = [texto]

    elif isinstance(texto, list):
        textos = texto

    else:
        raise TypeError(
            "texto debe ser una cadena o una lista de cadenas."
        )

    if len(textos) == 0:
        raise ValueError(
            "La lista de textos no puede estar vacía."
        )

    if not all(isinstance(elemento, str) for elemento in textos):
        raise TypeError(
            "Todos los elementos de la lista deben ser cadenas."
        )

    if max_length <= 0:
        raise ValueError(
            "max_length debe ser mayor que cero."
        )

    tokens = tokenizer(
        textos,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    tokens = {
        key: value.to(device)
        for key, value in tokens.items()
    }

    modelo.eval()

    outputs = modelo(**tokens)

    return outputs, tokens


def crear_clasificador_multiclase(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_hidden_layers: int = 1,
    activation: str = "gelu",
    normalization: str | None = "layernorm",
    dropout: float = 0.30,
    device: torch.device | None = None,
) -> nn.Sequential:
    """
    Construye dinámicamente una red neuronal multiclase.

    Arquitectura general
    --------------------
    Cuando num_hidden_layers >= 1:

        Linear(input_dim, hidden_dim)
        [Normalización]
        [Activación]
        Dropout

        Linear(hidden_dim, hidden_dim)
        [Normalización]
        [Activación]
        Dropout

        ...

        Linear(hidden_dim, num_classes)

    Cuando num_hidden_layers == 0:

        Linear(input_dim, num_classes)

    Parámetros
    ----------
    input_dim:
        Dimensión del vector producido por el encoder.

        Ejemplo para microsoft/deberta-v3-large:

            input_dim = 1024

    hidden_dim:
        Número de neuronas de cada capa oculta.

        Ejemplos:

            128, 256, 512

        Este parámetro no se utiliza cuando:

            num_hidden_layers = 0

    num_classes:
        Número de categorías del problema.

        La capa final generará un logit por clase.

    num_hidden_layers:
        Número de capas ocultas del clasificador.

        Valores posibles:

            0:
                Clasificador lineal directo.

                Linear(input_dim, num_classes)

            1:
                Una capa oculta.

                Linear(input_dim, hidden_dim)
                ...
                Linear(hidden_dim, num_classes)

            2 o más:
                Agrega capas adicionales con dimensiones:

                Linear(hidden_dim, hidden_dim)

    activation:
        Función de activación utilizada después de cada
        capa lineal oculta.

        Valores admitidos:

            "gelu"
            "relu"
            "silu"
            "tanh"
            "leaky_relu"

    normalization:
        Tipo de normalización aplicado después de cada
        capa lineal oculta.

        Valores admitidos:

            "layernorm":
                Aplica nn.LayerNorm(hidden_dim).

            "batchnorm":
                Aplica nn.BatchNorm1d(hidden_dim).

            None o "none":
                No aplica normalización.

    dropout:
        Probabilidad de desactivar activaciones durante
        el entrenamiento.

        Debe encontrarse en el intervalo:

            0 <= dropout < 1

    device:
        Dispositivo al que se moverá la red.

        Ejemplos:

            torch.device("cuda")
            torch.device("cpu")

    Retorna
    -------
    nn.Sequential:
        Clasificador multiclase construido dinámicamente.
    """

    # ========================================================
    # 1. VALIDACIONES
    # ========================================================

    if input_dim <= 0:
        raise ValueError(
            "input_dim debe ser mayor que cero."
        )

    if hidden_dim <= 0:
        raise ValueError(
            "hidden_dim debe ser mayor que cero."
        )

    if num_classes < 2:
        raise ValueError(
            "num_classes debe ser igual o mayor que dos."
        )

    if num_hidden_layers < 0:
        raise ValueError(
            "num_hidden_layers no puede ser negativo."
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "dropout debe encontrarse en el intervalo [0, 1)."
        )

    activation = activation.lower()

    if normalization is not None:
        normalization = normalization.lower()

    # ========================================================
    # 2. SELECCIONAR LA FUNCIÓN DE ACTIVACIÓN
    # ========================================================

    activation_functions = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }

    if activation not in activation_functions:
        raise ValueError(
            f"Función de activación no válida: {activation}. "
            f"Opciones disponibles: "
            f"{list(activation_functions.keys())}"
        )

    # Se guarda la clase de la función de activación.
    # Se creará una instancia nueva en cada capa oculta.
    activation_class = activation_functions[activation]

    # ========================================================
    # 3. VALIDAR EL TIPO DE NORMALIZACIÓN
    # ========================================================

    valid_normalizations = {
        "layernorm",
        "batchnorm",
        "none",
        None,
    }

    if normalization not in valid_normalizations:
        raise ValueError(
            "normalization debe ser 'layernorm', "
            "'batchnorm', 'none' o None."
        )

    # ========================================================
    # 4. CONSTRUIR LAS CAPAS
    # ========================================================

    layers = []

    # --------------------------------------------------------
    # CASO A: CLASIFICADOR LINEAL SIN CAPAS OCULTAS
    # --------------------------------------------------------

    if num_hidden_layers == 0:

        layers.append(
            nn.Linear(
                in_features=input_dim,
                out_features=num_classes,
            )
        )

    # --------------------------------------------------------
    # CASO B: CLASIFICADOR CON UNA O MÁS CAPAS OCULTAS
    # --------------------------------------------------------

    else:

        # Dimensión de entrada de la primera capa.
        current_input_dim = input_dim

        for layer_index in range(num_hidden_layers):

            # Transformación lineal.
            #
            # Primera capa:
            #     input_dim -> hidden_dim
            #
            # Capas posteriores:
            #     hidden_dim -> hidden_dim
            layers.append(
                nn.Linear(
                    in_features=current_input_dim,
                    out_features=hidden_dim,
                )
            )

            # Normalización opcional.
            if normalization == "layernorm":

                layers.append(
                    nn.LayerNorm(hidden_dim)
                )

            elif normalization == "batchnorm":

                layers.append(
                    nn.BatchNorm1d(hidden_dim)
                )

            # Agregar una instancia independiente
            # de la función de activación.
            layers.append(
                activation_class()
            )

            # Regularización.
            #
            # Si dropout = 0, se omite la capa.
            if dropout > 0:

                layers.append(
                    nn.Dropout(p=dropout)
                )

            # Después de la primera capa, todas las capas
            # reciben vectores de dimensión hidden_dim.
            current_input_dim = hidden_dim

        # Capa final.
        #
        # Genera un logit por clase:
        #
        # [batch_size, hidden_dim]
        #              ↓
        # [batch_size, num_classes]
        layers.append(
            nn.Linear(
                in_features=hidden_dim,
                out_features=num_classes,
            )
        )

    # ========================================================
    # 5. CREAR EL MODELO SECUENCIAL
    # ========================================================

    clasificador = nn.Sequential(*layers)

    # ========================================================
    # 6. MOVER EL MODELO AL DISPOSITIVO
    # ========================================================

    if device is not None:
        clasificador = clasificador.to(device)

    return clasificador


def crear_clasificador_binario(
    input_dim: int,
    hidden_dim: int = 256,
    num_hidden_layers: int = 1,
    activation: str = "gelu",
    normalization: str | None = "layernorm",
    dropout: float = 0.30,
    device: torch.device | None = None,
) -> nn.Sequential:
    """
    Construye dinámicamente una red neuronal para clasificación binaria.

    Arquitectura general
    --------------------
    Cuando num_hidden_layers >= 1:

        Linear(input_dim, hidden_dim)
        [Normalización]
        [Activación]
        Dropout

        Linear(hidden_dim, hidden_dim)
        [Normalización]
        [Activación]
        Dropout

        ...

        Linear(hidden_dim, 1)

    Cuando num_hidden_layers == 0:

        Linear(input_dim, 1)

    Parámetros
    ----------
    input_dim:
        Dimensión del vector generado por el encoder.

        Ejemplo para microsoft/deberta-v3-large:

            input_dim = 1024

    hidden_dim:
        Número de neuronas de cada capa oculta.

        Ejemplos:

            128, 256, 512

        No se utiliza cuando:

            num_hidden_layers = 0

    num_hidden_layers:
        Número de capas ocultas.

        Valores posibles:

            0:
                Clasificador lineal sin capas ocultas.

            1:
                Una capa oculta.

            2 o más:
                Varias capas ocultas.

    activation:
        Función de activación aplicada después de cada
        capa lineal oculta.

        Valores admitidos:

            "gelu"
            "relu"
            "silu"
            "tanh"
            "leaky_relu"

    normalization:
        Tipo de normalización aplicado después de cada
        capa lineal oculta.

        Valores admitidos:

            "layernorm"
            "batchnorm"
            "none"
            None

    dropout:
        Probabilidad de desactivar activaciones durante
        el entrenamiento.

        Debe cumplir:

            0 <= dropout < 1

    device:
        Dispositivo donde se almacenará el clasificador.

        Ejemplos:

            torch.device("cuda")
            torch.device("cpu")

    Retorna
    -------
    nn.Sequential:
        Clasificador binario que genera un único logit
        por observación.
    """

    # ========================================================
    # 1. VALIDACIONES
    # ========================================================

    if input_dim <= 0:
        raise ValueError(
            "input_dim debe ser mayor que cero."
        )

    if hidden_dim <= 0:
        raise ValueError(
            "hidden_dim debe ser mayor que cero."
        )

    if num_hidden_layers < 0:
        raise ValueError(
            "num_hidden_layers no puede ser negativo."
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "dropout debe encontrarse en el intervalo [0, 1)."
        )

    activation = activation.lower()

    if normalization is not None:
        normalization = normalization.lower()

    # ========================================================
    # 2. FUNCIONES DE ACTIVACIÓN DISPONIBLES
    # ========================================================

    activation_functions = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }

    if activation not in activation_functions:
        raise ValueError(
            f"Función de activación no válida: {activation}. "
            f"Opciones disponibles: "
            f"{list(activation_functions.keys())}"
        )

    activation_class = activation_functions[activation]

    # ========================================================
    # 3. TIPOS DE NORMALIZACIÓN DISPONIBLES
    # ========================================================

    valid_normalizations = {
        "layernorm",
        "batchnorm",
        "none",
        None,
    }

    if normalization not in valid_normalizations:
        raise ValueError(
            "normalization debe ser 'layernorm', "
            "'batchnorm', 'none' o None."
        )

    # ========================================================
    # 4. CONSTRUCCIÓN DINÁMICA DE LA ARQUITECTURA
    # ========================================================

    layers = []

    # --------------------------------------------------------
    # CASO A: SIN CAPAS OCULTAS
    # --------------------------------------------------------

    if num_hidden_layers == 0:

        # Clasificador lineal directo:
        #
        # [batch_size, input_dim]
        #              ↓
        # [batch_size, 1]
        layers.append(
            nn.Linear(
                in_features=input_dim,
                out_features=1,
            )
        )

    # --------------------------------------------------------
    # CASO B: UNA O MÁS CAPAS OCULTAS
    # --------------------------------------------------------

    else:

        current_input_dim = input_dim

        for layer_index in range(num_hidden_layers):

            # Primera capa:
            #
            # input_dim -> hidden_dim
            #
            # Capas posteriores:
            #
            # hidden_dim -> hidden_dim
            layers.append(
                nn.Linear(
                    in_features=current_input_dim,
                    out_features=hidden_dim,
                )
            )

            # Normalización opcional.
            if normalization == "layernorm":

                layers.append(
                    nn.LayerNorm(hidden_dim)
                )

            elif normalization == "batchnorm":

                layers.append(
                    nn.BatchNorm1d(hidden_dim)
                )

            # Función de activación.
            layers.append(
                activation_class()
            )

            # Regularización.
            if dropout > 0:

                layers.append(
                    nn.Dropout(p=dropout)
                )

            # Las siguientes capas reciben hidden_dim.
            current_input_dim = hidden_dim

        # Capa final binaria.
        #
        # [batch_size, hidden_dim]
        #              ↓
        # [batch_size, 1]
        layers.append(
            nn.Linear(
                in_features=hidden_dim,
                out_features=1,
            )
        )

    # ========================================================
    # 5. CREAR EL MODELO SECUENCIAL
    # ========================================================

    clasificador = nn.Sequential(*layers)

    # ========================================================
    # 6. MOVER EL MODELO AL DISPOSITIVO
    # ========================================================

    if device is not None:
        clasificador = clasificador.to(device)

    return clasificador



def preparar_dataset_vectores(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
    tokenizer,
    modelo_bert: torch.nn.Module,
    device: torch.device,
    pooling: str = "mean",
    max_length: int = 128,
    embedding_batch_size: int = 8,
    label_encoder: LabelEncoder | None = None,
) -> tuple[TensorDataset, LabelEncoder, torch.Tensor]:
    """
    Convierte un DataFrame de textos y etiquetas en un TensorDataset
    listo para utilizarse con DataLoader.

    El modelo tipo BERT se utiliza como extractor de características
    y permanece congelado.

    Parámetros
    ----------
    dataframe:
        DataFrame con una columna de textos y una columna de etiquetas.

    text_column:
        Nombre de la columna que contiene los textos.

    label_column:
        Nombre de la columna que contiene las clases.

    tokenizer:
        Tokenizador correspondiente al modelo BERT/DeBERTa.

    modelo_bert:
        Encoder preentrenado, por ejemplo:
        microsoft/deberta-v3-large.

    device:
        Dispositivo donde se ejecutará el encoder:
        torch.device("cuda") o torch.device("cpu").

    pooling:
        Método para obtener un vector por texto:

        "cls":
            Toma el vector del primer token.

        "mean":
            Promedia los vectores de los tokens válidos.

        "max":
            Selecciona el máximo por dimensión entre los tokens válidos.

    max_length:
        Longitud máxima de tokenización.

    embedding_batch_size:
        Cantidad de textos procesados simultáneamente por el encoder.

    label_encoder:
        Codificador de etiquetas previamente ajustado.

        Se debe proporcionar al transformar validación o prueba para
        mantener la misma correspondencia entre clases e identificadores.

        Si es None, se crea y ajusta uno nuevo.

    Retorna
    -------
    tensor_dataset:
        TensorDataset que contiene:

            vectores:  [numero_ejemplos, hidden_size]
            etiquetas: [numero_ejemplos]

    label_encoder:
        LabelEncoder utilizado para transformar las etiquetas.

    vectores:
        Tensor completo con las representaciones generadas.
    """

    # --------------------------------------------------------
    # 1. Validaciones
    # --------------------------------------------------------


    if pooling not in {"cls", "mean", "max"}:
        raise ValueError(
            "pooling debe ser 'cls', 'mean' o 'max'."
        )

    # --------------------------------------------------------
    # 3. Codificar las etiquetas
    # --------------------------------------------------------
    datos=dataframe
    etiquetas_texto = (
        datos[label_column]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    if label_encoder is None:
        label_encoder = LabelEncoder()

        etiquetas_numericas = label_encoder.fit_transform(
            etiquetas_texto
        )

    else:
        try:
            etiquetas_numericas = label_encoder.transform(
                etiquetas_texto
            )

        except ValueError as error:
            raise ValueError(
                "El DataFrame contiene una etiqueta que no fue "
                "registrada en el LabelEncoder."
            ) from error

    etiquetas_tensor = torch.tensor(
        etiquetas_numericas,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # 4. Congelar y preparar el encoder
    # --------------------------------------------------------
    textos = datos[text_column].tolist()

    vectores_generados = []

    # --------------------------------------------------------
    # 5. Procesar textos por lotes
    # --------------------------------------------------------

    for inicio in tqdm(
        range(0, len(textos), embedding_batch_size),
        desc=f"Generando vectores ({pooling})",
    ):
        fin = inicio + embedding_batch_size

        lote_textos = textos[inicio:fin]

        tokens = tokenizer(
            lote_textos,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        tokens = {
            nombre: tensor.to(device)
            for nombre, tensor in tokens.items()
        }

        # No se calculan gradientes porque el encoder está congelado.
        with torch.inference_mode():

            outputs = modelo_bert(**tokens)

            last_hidden_state = outputs.last_hidden_state

            # ------------------------------------------------
            # CLS pooling
            # ------------------------------------------------

            if pooling == "cls":

                vectores_lote = (
                    last_hidden_state[:, 0, :]
                )

            # ------------------------------------------------
            # Mean pooling
            # ------------------------------------------------

            elif pooling == "mean":

                attention_mask = tokens[
                    "attention_mask"
                ]

                mascara_expandida = (
                    attention_mask
                    .unsqueeze(-1)
                    .to(last_hidden_state.dtype)
                )

                suma_vectores = (
                    last_hidden_state
                    * mascara_expandida
                ).sum(dim=1)

                numero_tokens_validos = (
                    mascara_expandida
                    .sum(dim=1)
                    .clamp(min=1.0)
                )

                vectores_lote = (
                    suma_vectores
                    / numero_tokens_validos
                )

            # ------------------------------------------------
            # Max pooling
            # ------------------------------------------------

            else:

                attention_mask = tokens[
                    "attention_mask"
                ]

                mascara_expandida = (
                    attention_mask
                    .unsqueeze(-1)
                    .bool()
                )

                valor_minimo = torch.finfo(
                    last_hidden_state.dtype
                ).min

                vectores_enmascarados = (
                    last_hidden_state.masked_fill(
                        ~mascara_expandida,
                        valor_minimo,
                    )
                )

                vectores_lote = (
                    vectores_enmascarados
                    .max(dim=1)
                    .values
                )

        # Entrenaremos la cabeza clasificadora en float32.
        vectores_lote = (
            vectores_lote
            .float()
            .cpu()
        )

        vectores_generados.append(
            vectores_lote
        )

    # --------------------------------------------------------
    # 6. Unir todos los lotes
    # --------------------------------------------------------

    vectores = torch.cat(
        vectores_generados,
        dim=0,
    )

    if vectores.shape[0] != etiquetas_tensor.shape[0]:
        raise RuntimeError(
            "La cantidad de vectores no coincide con "
            "la cantidad de etiquetas."
        )

    # --------------------------------------------------------
    # 7. Crear TensorDataset
    # --------------------------------------------------------

    tensor_dataset = TensorDataset(
        vectores,
        etiquetas_tensor,
    )

    return tensor_dataset
    
    
