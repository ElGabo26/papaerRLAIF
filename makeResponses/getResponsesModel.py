import pandas as pd
import os
from tqdm import tqdm
from time import time
from  tools import  makeResponse, testModel
from metrics import medir_recursos
from loadAlginModelTools import cargar_modelo, generar_respuestas_batch

REPETITIONS=1
MODEL_ROUTE="/workspace/models"
RUTAOUTPUT="/workspace/papaerRLAIF/makeResponses/responses"
base=pd.read_csv("/workspace/papaerRLAIF/makeResponses/promptBases/finalPromptBases/elementary_math_prompts_1200.csv")
prompts=base['prompt'].values.tolist()
total=len(prompts)




def getresults(token, model,x):
    t0=time()
    r=makeResponse(token,model,x)
    #despues=medir_recursos()
    t1=time()
    barra.update(1)
    return x,r,t1-t0

#models=['/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B', '/workspace/models/Qwen2.5-1.5B-Instruct']
models=os.listdir(MODEL_ROUTE)
models=[f"{MODEL_ROUTE}/{x}" if x != "deberta-v3-large" else None for x in models]


for ruta in models:
    if ruta:
        print("MODELO: ", ruta)
        for i in range(REPETITIONS): 
            print(f"repeticion_{i+1}")
            columnas=['prompt',f'response',f'tiempo_{i+1}']  
            model, tokenizer = testModel(ruta)
            resultado=generar_respuestas_batch(model,tokenizer,
                                       prompts,batch_size=20,max_new_tokens=200,num_responses=1)
            r=pd.DataFrame(columns=columnas,data=resultado)
            r['prompt_id']=prompts['prompt_id']
            name='result'+ruta.split('/')[-1]
            r.to_csv(f"{RUTAOUTPUT}/{name}_{i+1}.csv")
            print(f"RESPUESTAS: {i+1} REALIZADAS")
    else:
        pass


