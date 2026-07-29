import pandas as pd
import tqdm
import os
from loadAlginModelTools import cargar_modelo, generar_respuestas_batch

PROMPTS_CSV = "/workspace/papaerRLAIF/makeResponses/promptBases/finalPromptBases/elementary_math_prompts_1200.csv"
prompts_df = pd.read_csv(PROMPTS_CSV)
prompts_df=prompts_df.sample(200, random_state=42)
DPO_MODELS_PATH="/workspace/adaptedModels/DPO"
PPO_MODELS_PATH="/workspace/adaptedModels/PPO/finalPPOModels"
RESPONSES_PATH="/workspace/papaerRLAIF/makeResponses/responses"

prompts=prompts_df["prompt"].tolist()

#respuestas  DPO 
'''
dpoRoutes=os.listdir(DPO_MODELS_PATH)
for dpoRoute in dpoRoutes:
    responses=[]
    model_path = os.path.join(DPO_MODELS_PATH, dpoRoute)
    model, tokenizer = cargar_modelo(model_path)
    responses=generar_respuestas_batch(model,tokenizer,
                                       prompts,batch_size=120,max_new_tokens=200,num_responses=1)
            
    result_df = pd.DataFrame({"prompt": prompts, "response": responses})
    result_df['algin']="DPO"
    result_df['model']=dpoRoute
    result_df.to_csv(os.path.join(RESPONSES_PATH, f"responses_DPO_{dpoRoute}.csv"), index=False)
    print(f"Respuestas generadas y guardadas para el modelo DPO: {dpoRoute}")
#respuestas PPO
'''
ppoRoutes=os.listdir(PPO_MODELS_PATH)
for ppoRoute in ppoRoutes:
    model_path = os.path.join(PPO_MODELS_PATH, ppoRoute)
    model, tokenizer = cargar_modelo(model_path)
    responses=[]
    responses=generar_respuestas_batch(model,tokenizer,
                                       prompts,batch_size=120,max_new_tokens=200,num_responses=1)
    result_df = pd.DataFrame({"prompt": prompts, "response": responses})
    result_df['algin']="PPO"
    result_df['model']=ppoRoute
    result_df.to_csv(os.path.join(RESPONSES_PATH, f"responses_PPO_{ppoRoute}.csv"), index=False)
    print(f"Respuestas generadas y guardadas para el modelo PPO: {ppoRoute}")