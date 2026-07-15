import pandas as pd
from time import time
from  tools import  makeResponse, testModel
from metrics import medir_recursos

RUTA=input("inserte ruta  del  modelo: ")
REPETITIONS=int(input("Inserte Cantidad de  respuestas: "))
RUTAOUTPUT="/workspace/papaerRLAIF/codPreferences/bases"
base=pd.read_csv("codPreferences/prompts.csv")
prompts=base.sample(frac=0.025, random_state=42)
prompts1=prompts['prompt'].values

token, model= testModel(RUTA)
result=prompts.copy()

for i in range(REPETITIONS): 
    response=[]
    timeR=[]
    ram=[]
    gpu=[]
    c=0
    for p in prompts1:
        t0=time()
        antes =medir_recursos()
        r=makeResponse(token,model,p,0.25)
        despues=medir_recursos()
        t1=time()
        tr=t1-t0
        timeR.append(tr)
        response.append(r)
        ram.append(despues['ram_mb']-antes['ram_mb'])
        gpu.append(despues['gpu_mb']-antes['gpu_mb'])
        print(len(prompts)-c)
        c+=1

    result[f'response_{i+1}']=response
    result[f'time_{i+1}']=timeR
    result[f'ram_mb_{i+1}']=ram
    result[f'gpu_mb_{i+1}']=gpu
    print(f"RESPUESTAS: {i+1} REALIZADAS")


name='result'+RUTA.split('/')[-1]
result.to_csv(f"{RUTAOUTPUT}/{name}.csv")


