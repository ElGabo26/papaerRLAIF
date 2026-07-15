import pandas as pd
from tools import *

#lista de  modelos
modelsName=["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
"Qwen/Qwen2.5-1.5B-Instruct",
"meta-llama/Llama-3.2-1B-Instruct"]
LOCALROOT="/workspace/models"

modelsroot=[]
models={}
for  m  in  modelsName:
    ruta =dowloadModel(m,LOCALROOT)
    print(ruta)
    #modelsroot.append(modelsroot)
    #t, model =testModel(ruta)
    #models[m]=(t,m)
    



