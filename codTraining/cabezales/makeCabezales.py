import torch
import torch.nn as nn
import joblib
import pandas as pd

from torch.utils.data import TensorDataset
torch.serialization.add_safe_globals([TensorDataset])


from tools import (crear_clasificador_binario, 
                   crear_clasificador_multiclase)

from trainingTools import (
    makeDivision, createLoaders, 
    train_eval_multiclass, train_eval_binary
)

from evaluationTools import(
    eval_binary, eval_multiclass
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LEARNING_RATE= 3e-4
DATA_ROUTE="/workspace/papaerRLAIF/codTraining/cabezales/vectorBases"
OUTPUT_MODEL="/workspace/adaptedModels/cabezales"
OUTPUT_EVAL="/workspace/papaerRLAIF/codTraining/cabezales/"
PARAMS_ROUTE="/workspace/papaerRLAIF/codTraining/cabezales/finalCabezalParams.joblib"
SEED=42
NUM_CLASSES=7
NUM_EPOCHS=20
MIN_DELTA=0.001

#obtenemos los  parametros  de  entrenamiento
params=joblib.load(PARAMS_ROUTE)

config_binary_model=[
    'input_dim',
    'hidden_dim',
    'num_hidden_layers',
    'activation',
    'normalization',
    'dropout',
    'device',
]

config_multiclass_model=[
    'input_dim',
    'hidden_dim',
    'num_classes',
    'num_hidden_layers',
    'activation',
    'normalization',
    'dropout',
    'device',
]

class_names=['addition-subtraction', 'subtraction', 'challenge',
       'division-multiplication', 'algebra', 'otro', 'geometry']

result_multi=[]
result_binary=[]
#funcion de entrenamiento
def makeRed_binario(config:dict, DEVICE:str, seed:int,trainL,testL):
    criterio = nn.BCEWithLogitsLoss()
    clasificador=crear_clasificador_binario(**config)
    optimizador = torch.optim.AdamW(
    clasificador.parameters(),
    lr=LEARNING_RATE)

    model, data=train_eval_binary(DEVICE,20, clasificador,criterio,
                                    optimizador, 
                                    trainL,testL,umbral=0.5,patience=3)
    return model

def makeRed_multi(config:dict, DEVICE:str, seed:int,trainL,testL):
    criterio = nn.CrossEntropyLoss()
    clasificador=crear_clasificador_multiclase(**config)
    optimizador = torch.optim.AdamW(
    clasificador.parameters(),
    lr=LEARNING_RATE)

    model, data=train_eval_multiclass(DEVICE,NUM_EPOCHS, clasificador,criterio,
                                    optimizador, 
                                    trainL,testL,5,MIN_DELTA)
    
    return model



for i,j in params.items():
    param=j.copy()
    param['device']=DEVICE
    param['input_dim']=1024
    name_base=f"{i}_{param['pooling']}.pt"
    data = torch.load(
            f"{DATA_ROUTE}/{name_base}",
            map_location="cpu",
            weights_only=True)
    
    input_dim = data.tensors[0].shape[1]
    train, test, eval =makeDivision(data,0.30,SEED)
    trainL, testL , evalL=createLoaders(256,train,test,eval)
    if i=='skill':
        param1=param.copy()
        param1['num_classes']=NUM_CLASSES
        config={x:param1[x] for x in config_multiclass_model}
        #entranmos el  modelo
        model=makeRed_multi(config,DEVICE,SEED,trainL,testL)
        print('MODELO ENTRENADO')
        #evaluamos el modelo
        criterio = nn.CrossEntropyLoss()
        resultadoTest=eval_multiclass(model,testL,DEVICE,criterio,class_names)
        resultadoVal=eval_multiclass(model,evalL,DEVICE,criterio,class_names)
        print('MODELO EVALUADO')
        resultadoTest['type']='test'
        resultadoTest['type']='val'
        result_multi.append(resultadoTest)
        result_multi.append(resultadoVal)
    else:
        config={x:param[x] for x in config_binary_model}
        #entranmos el  modelo
        model=makeRed_binario(config, DEVICE,SEED,trainL, testL)
        print('MODELO ENTRENADO')
        #evaluamos el  modelo
        criterio = nn.BCEWithLogitsLoss()
        resultadoTest=eval_binary(model,testL,DEVICE,criterio)
        resultadoVal=eval_binary(model,evalL,DEVICE,criterio)
        print('MODELO EVALUADO')
        resultadoTest['type']='test'
        resultadoTest['type']='val'
        result_binary.append(resultadoTest)
        result_binary.append(resultadoVal)
    torch.save(model,f"{OUTPUT_MODEL}/{i}.pt")
    print(f'CABEZAL {i} GUARDADO')

resultBinary=pd.DataFrame(result_binary)
resultMulti=pd.DataFrame(result_multi)

resultBinary.to_csv(f"{OUTPUT_EVAL}/binari_models.csv")
resultMulti.to_csv(f"{OUTPUT_EVAL}/multi_models.csv")
print('EVAL GUARDADO') 


