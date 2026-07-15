import pandas as pd
from  tools import  makeResponse, testModel

RUTA='/workspace/models/Qwen2.5-1.5B-Instruct'
base=pd.read_csv("codPreferences/prompts.csv")
prompts=base.sample(300, random_state=42)
prompts1=prompts['prompt'].values

token, model= testModel(RUTA)

response=[]
c=0
for p in prompts1:
    r=makeResponse(token,model,p,0.25)
    print(len(prompts)-c)
    c+=1
    

result=prompts.copy()

result['response']=response

name='result'+RUTA.split('/')[-1]
result.to_csv(f"{name}.csv")


