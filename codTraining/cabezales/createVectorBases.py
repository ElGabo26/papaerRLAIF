from os import listdir
import joblib
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
DATABASES_ROUTE="/workspace/papaerRLAIF/codTraining/cabezales/trainingCabezales"
OUTPUT_ROUTE="/workspace/papaerRLAIF/codTraining/cabezales/vectorBases"
ENCODER_ROUTE="/workspace/papaerRLAIF/codTraining/cabezales/encoders"
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

dataroutes=listdir(DATABASES_ROUTE)
poolins=['cls','mean', 'max']

for base in dataroutes:
    db=pd.read_csv(f"{DATABASES_ROUTE}/{base}", index_col=0)
    if db.shape[1]!=2 or list(db.columns)!=["text","labels"]:
        raise ValueError(
            "la base  de  datos no  tiene  las columnas necesarias"
        )
    name=base.split('.')[0]
    encoder=LabelEncoder()
    encoder.fit(db['labels'])
    joblib.dump(encoder,f"{ENCODER_ROUTE}/{name}.joblib")
    print(f"encoder {base} guarrdado")
    
    for pool in poolins:
        print(f"base polling {pool}")
        tensor_dataset=preparar_dataset_vectores(
            db,'text','labels', tokenizer,modelo_bert,
            DEVICE,pool,256,8,encoder)

        indices = np.arange(
            len(tensor_dataset)
                            )

        torch.save(
            tensor_dataset,
            f"{OUTPUT_ROUTE}/{name}_{pool}.pt"
            )
        print(f"BASE {name}_{pool} CREADA")
