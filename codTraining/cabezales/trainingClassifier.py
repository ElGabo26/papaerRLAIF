import torch
import torch.nn as nn
from os import listdir
from pandas import DataFrame, concat
from tools import (crear_clasificador_binario, 
                   crear_clasificador_multiclase)

from trainingTools import (
    makeDivision, createLoaders, 
    train_eval_multiclass, train_eval_binary
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LEARNING_RATE= 3e-4
DATA_ROUTE="/workspace/papaerRLAIF/codTraining/cabezales/vectorBases"
OUTPUT_MODEL="/workspace/papaerRLAIF/codTraining/cabezales/models"

bases=listdir(DATA_ROUTE)
#bases  de  claridad
bases=[x for x in  bases if 'claridad']


seeds=[ 42,    123,    2024]
num_hidden_layers_options = [0, 1, 2, 3]
hidden_dim_options = [128, 256, 512]
activation_options = [ "gelu",    "relu",    "silu"]
normalization_options = [    None,    "layernorm",    "batchnorm"]
dropout_options = [    0.0,    0.1,    0.3,    0.5,]
poolins=['cls','mean', 'max']
# Función de pérdida para clasificación binaria.
criterio = nn.BCEWithLogitsLoss()
# Optimizador.

result=DataFrame()
for db in bases:
    seed=seeds[0]
    print(seed)
    print("tipo pooling:", db.split('_')[1].split('.')[0])
    datos = torch.load(
        f"{DATA_ROUTE}/{db}",
        map_location="cpu",
        weights_only=True)
    input_dim = datos.tensors[0].shape[1]
    print(f"INPUT  DE DIMENSION: {input_dim}")
    train, test, eval =makeDivision(datos,0.30,seed)
    trainL, testL , evalL=createLoaders(16,train,test,eval)
    
    clasificador=crear_clasificador_binario(input_dim, 256,1, 'gelu','layernorm',0.3,DEVICE)
    
    optimizador = torch.optim.AdamW(
    clasificador.parameters(),
    lr=LEARNING_RATE)
    
    model, data=train_eval_binary(DEVICE,20, clasificador,criterio,optimizador, trainL,testL)
    data['pooling']=db.split('_')[1].split('.')[0]
    data['seed']=seed
    data['hidden_dim']=256
    data['hidden_layers']=1
    data['norm']='layernorm'
    data['dropout']=0.3
    print("MODELO ENTRENADO")
    torch.save(
        model, f"{OUTPUT_MODEL}/testmodel_claridad_{data['pooling'].values[0]}.pt"    
    )
    print("MODELO GUARDADO")
    resultado=concat([data,result])

resultado.to(f"{OUTPUT_MODEL}/testmodel_metrics_claridad.csv")
