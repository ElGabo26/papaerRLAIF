import pandas as pd
from  tools import  makeResponse, testModel

RUTA='/workspace/models/Qwen2.5-1.5B-Instruct'
base=pd.read_csv("codPreferences/prompts.csv")
prompts=base['prompt'].values

token, model= testModel(RUTA)

response=list(map(lambda x: makeResponse(token, model ,x,0.25),
    prompts))

result=base.copy()

result['response']=response

name='result'+RUTA.split('/')[-1]
result.to_csv(f"{name}.csv")


