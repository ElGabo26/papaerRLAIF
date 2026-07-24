import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
torch.serialization.add_safe_globals([TensorDataset])
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
bases=[x for x in  bases if 'claridad' in x]



seeds=[ 42,    123,    2024]
num_hidden_layers_options = [0, 1, 2, 3]
hidden_dim_options = [128, 256, 512]
activation_options = [ "gelu",    "relu",    "silu"]
normalization_options = [    None,    "layernorm",    "batchnorm"]
dropout_options = [    0.0,    0.1,    0.3,    0.5]

parametros={
    'seed':seeds,
    'num_hidden_layer':num_hidden_layers_options,
    'hidden_dim':hidden_dim_options,
    'activation_options':activation_options,
    'normalization':normalization_options,
    'dropout':dropout_options    
}

# Función de pérdida para clasificación binaria.
criterio = nn.BCEWithLogitsLoss()
# Optimizador.

result=[]
for db in bases:
    pooling=db.split('_')[1].split('.')[0]
    print(pooling)
    for seed in seeds:
        print(seed)
        datos = torch.load(
            f"{DATA_ROUTE}/{db}",
            map_location="cpu",
            weights_only=True)
        input_dim = datos.tensors[0].shape[1]
        train, test, eval =makeDivision(datos,0.30,seed)
        trainL, testL , evalL=createLoaders(16,train,test,eval)
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
            name=f'model_claridad_{pooling}_{seed}_{hidden_dim}.pt'
            torch.save(
                model, f"{OUTPUT_MODEL}/{name}"    
            )
            result.append(data)
            print("MODELO GUARDADO")
        

resultado=concat(result)
resultado.to_csv(f"{OUTPUT_MODEL}/metadata_claridad_hidden_dim.csv")
