import pandas as pd
from .tools import *

#lista de  modelos
modelsName=["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 1.5B",
"Qwen/Qwen2.5-1.5B-Instruct 1.5B",
"meta-llama/Llama-3.2-1B-Instruct"]
LOCALROOT="/"

modelsroot=[]
models={}
for  m  in  modelsName:
    ruta =dowloadModel(m,LOCALROOT)
    modelsroot.append(modelsroot)
    t, model =testModel(modelsroot)
    models[m]=(t,m)
    



