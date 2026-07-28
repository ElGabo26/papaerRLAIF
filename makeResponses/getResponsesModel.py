import pandas as pd
import os
from tqdm import tqdm
from time import time
from  tools import  makeResponse, testModel
from metrics import medir_recursos

REPETITIONS=1
MODEL_ROUTE="/workspace/models"
RUTAOUTPUT="/workspace/papaerRLAIF/makeResponses/responses"
base=pd.read_csv("/workspace/papaerRLAIF/makeResponses/promptBases/finalPromptBases/elementary_math_prompts_1200.csv")
prompts=base.sample(frac=1.0, random_state=42)
prompts1=prompts['prompt'].values
total=len(prompts1)




def getresults(token, model,x):
    t0=time()
    r=makeResponse(token,model,x,0.25)
    despues=medir_recursos()
    t1=time()
    barra.update(1)
    return x,r,t1-t0, despues['ram_mb'], despues['gpu_mb']

#models=['/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B', '/workspace/models/Qwen2.5-1.5B-Instruct']
models=os.listdir(MODEL_ROUTE)
models=[f"{MODEL_ROUTE}/{x}" if x != "deberta-v3-large" else None for x in models]
print(models)

for ruta in models:
    print("MODELO: ", ruta)
    token, model= testModel(ruta)    
    for i in range(REPETITIONS): 
        print(f"repeticion_{i+1}")
        columnas=['prompt',f'response',f'tiempo_{i+1}',f'ram_mb_{i+1}',f'gpu_mb_{i+1}']  
        with tqdm(total=total) as barra:
            resultado = list(map(
                lambda x: getresults(token, model,x),
                prompts1))
        r=pd.DataFrame(columns=columnas,data=resultado)
        name='result'+ruta.split('/')[-1]
        r.to_csv(f"{RUTAOUTPUT}/{name}_{i+1}.csv")
        print(f"RESPUESTAS: {i+1} REALIZADAS")


