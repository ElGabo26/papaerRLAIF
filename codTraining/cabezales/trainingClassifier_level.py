import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import product
from torch.utils.data import TensorDataset
torch.serialization.add_safe_globals([TensorDataset])
from os import listdir
from pandas import DataFrame, concat, read_csv
from tools import (crear_clasificador_binario, 
                   crear_clasificador_multiclase, batch_n)

from trainingTools import (
    makeDivision, createLoaders, 
    train_eval_multiclass, train_eval_binary, getbest
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LEARNING_RATE= 3e-4
DATA_ROUTE="/workspace/papaerRLAIF/codTraining/cabezales/vectorBases"
OUTPUT_MODEL="/workspace/papaerRLAIF/codTraining/cabezales/models"

bases=listdir(DATA_ROUTE)
#bases  de  level
bases=[x for x in  bases if 'level' in x]
print(bases)



seeds=[   123,    2024]
num_hidden_layers_options = [0, 1, 2, 3]
hidden_dim_options = [128, 256, 512]
activation_options = [ "gelu",    "relu",    "silu"]
normalization_options = [    None,    "layernorm",    "batchnorm"]
dropout_options = [    0.0,    0.1,    0.3,    0.5]


# Función de pérdida para clasificación binaria.
criterio = nn.BCEWithLogitsLoss()
# Optimizador.

#===============================================================
#DEFINIMOS POOLING
#===============================================================
result=[]
for db in bases:
    pooling=db.split('_')[1].split('.')[0]
    print(pooling)
    with tqdm(total=27) as barra:
        for seed in seeds:
            print(seed)
            datos = torch.load(
                f"{DATA_ROUTE}/{db}",
                map_location="cpu",
                weights_only=True)
            input_dim = datos.tensors[0].shape[1]
            train, test, eval =makeDivision(datos,0.30,seed)
            trainL, testL , evalL=createLoaders(256,train,test,eval)
            
            for hidden_dim in hidden_dim_options:
                config = {
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "num_hidden_layers": 1,
                    "activation": "gelu",
                    "normalization": "layernorm",
                    "dropout": 0.30,
                    "device": DEVICE,
                }

                clasificador=crear_clasificador_binario(**config)
                

                optimizador = torch.optim.AdamW(
                clasificador.parameters(),
                lr=LEARNING_RATE)

                model, data=train_eval_binary(DEVICE,20, clasificador,criterio,optimizador, trainL,testL,umbral=0.5,patience=3)
                
                config['seed']=seed
                data['pooling']=pooling
                for i,j in config.items():
                    data[i]=j
                
                result.append(data)
                barra.update(1)
                
                
                print(f"ENTRANAMIENTO COMPLETADO")
        

resultado=concat(result)
resultado.to_csv(f"{OUTPUT_MODEL}/metadata_level_pooling.csv")
print("resultado pooling guardado")

groupcols=['pooling']
metrics= ['train_f1','val_f1','train_accuracy','val_accuracy']
pooling=getbest(resultado,groupcols,metrics)
pooling=pooling['pooling']
print("POOLING DEFINIDO: ", pooling)

#===============================================================
#DEFINIMOS ARQUITECTURA RED
#===============================================================

#generamos  las distintas configuraciones
a=list(product(num_hidden_layers_options,hidden_dim_options,
               activation_options,normalization_options,
               dropout_options))
configNames = [ "num_hidden_layers",
                "hidden_dim",
                "activation",
                "normalization",
                "dropout"]
configs=list(map(
    lambda x: dict(zip(configNames,x))
    ,a))

print(DEVICE)

#funcion de  creacion  y  entrenamiento
def defRed(config:dict, DEVICE:str, seed:int,pooling:str):
    
    clasificador=crear_clasificador_binario(**config)
    optimizador = torch.optim.AdamW(
    clasificador.parameters(),
    lr=LEARNING_RATE)

    model, data=train_eval_binary(DEVICE,20, clasificador,criterio,
                                    optimizador, 
                                    trainL,testL,umbral=0.5,patience=3)
    
    config1=config.copy()
    config1['seed']=seed
    data['pooling']=pooling
    for i,j in config1.items():
        data[i]=j
    barra.update(1)
    
    return data

batchConfig=batch_n(configs,8)

#cargamos los  datos
print("DEFINIMOS ARQUITECTURA RED")
datos = torch.load(
            f"{DATA_ROUTE}/level_{pooling}.pt",
            map_location="cpu",
            weights_only=True)

input_dim=datos.tensors[0].shape[1]

configs=[x.update({"input_dim": input_dim, "device": DEVICE})
         for x in configs]

print("TOTAL CONFIGS ",len(configs))

finalBase=[]
for seed in seeds:
    print(seed)
    input_dim = datos.tensors[0].shape[1]
    train, test, eval =makeDivision(datos,0.30,seed)
    trainL, testL , evalL=createLoaders(512,train,test,eval)
    for batch in batchConfig:
        
        name=f"models_level_seed_{seed}_batch_{batchConfig.index(batch) +1}"
        with tqdm(total=len(batch)) as barra:
            
            resutlBatch=list(map(
                lambda x: defRed(x,DEVICE,seed, pooling)
            , batch))
        resultB=concat(resutlBatch)
        resultB.to_csv(f"{OUTPUT_MODEL}/{name}.csv")
        finalBase.append(resultB)
        print(name, "GUARDADO")

finalBase=concat(finalBase)
finalBase.to_csv(f"{OUTPUT_MODEL}/level_finalBase.csv")
print("finalBaseCreada")
        
