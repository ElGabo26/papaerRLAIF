

def preparar_dataset_vectores(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
    tokenizer,
    modelo_bert: torch.nn.Module,
    device: torch.device,
    pooling: str = "mean",
    max_length: int = 256,
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
    
