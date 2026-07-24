import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tools import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset, TensorDataset
from transformers import AutoModel, AutoTokenizer




# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_DOWLOAD= "microsoft/deberta-v3-large"
MODEL_ROUTE="/workspace/models/"
DATABASE_ROUTE="/workspace/papaerRLAIF/codTraining/cabezales/trainingCabezales/claridad.csv"
OUTPUT_ROUTE="/workspace/papaerRLAIF/codTraining/cabezales/vectorBases"
SEED =42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

rutaModelo="/workspace/models/deberta-v3-large"

# ============================================================
# CARGAR TOKENIZADOR Y MODELO
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
        rutaModelo,
        local_files_only=True,
        fix_mistral_regex=True
    )

model = AutoModel.from_pretrained(
        rutaModelo,
        local_files_only=True
    )

modelo_bert = model.to(DEVICE)
# Congelar todos los parámetros del modelo.
for parametro in modelo_bert.parameters():
    parametro.requires_grad = False
# Desactivar dropout y otras operaciones de entrenamiento.
modelo_bert.eval()
print("Dispositivo:", DEVICE)
print("Dimensión oculta:", modelo_bert.config.hidden_size)


# ============================================================
# CARGA  DE  DATOS
# ============================================================

dataframe=pd.read_csv(DATABASE_ROUTE, index_col=0)
if dataframe.shape[1]<2:
    raise ValueError(
        "El dataset  debe tener  dos columnas minimo")
#creamos  la base  tensorial
encoder=LabelEncoder()
encoder.fit(dataframe['labels'])

poolins=['cls','mean', 'max']
tensor_dataset=preparar_dataset_vectores(
    dataframe,'text','labels', tokenizer,modelo_bert,
    DEVICE,'cls',256,8,encoder)

indices = np.arange(
    len(tensor_dataset)
)

torch.save(
    tensor_dataset,
    f"{OUTPUT_ROUTE}/claridad.pt"
)





    