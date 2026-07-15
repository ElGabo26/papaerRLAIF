import pandas as pd
from tools import *

#lista de  modelos
modelsName=["meta-llama/Llama-3.2-1B-Instruct"]
LOCALROOT="/workspace/models"

modelsroot=[]
models={}
for  m  in  modelsName:
    ruta =dowloadModel(m,LOCALROOT)
    print(ruta)
    #modelsroot.append(modelsroot)
    #t, model =testModel(ruta)
    #models[m]=(t,m)
    



