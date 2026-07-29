import pandas as pd
import os
from loadAlginModelTools import cargar_modelo, generar_respuesta

PROMPTS_CSV = "/workspace/papaerRLAIF/makeResponses/promptBases/finalPromptBases/elementary_math_prompts_1200.csv"
prompts_df = pd.read_csv(PROMPTS_CSV)
DPO_MODELS_PATH="/workspace/adaptedModels/DPO"
PPO_MODELS_PATH="/workspace/adaptedModels/PPO"
RESPONSES_PATH="/workspace/papaerRLAIF/makeResponses/responses"

prompts=prompts_df["prompt"].tolist()

#respuestas  DPO 
dpoRoutes=os.listdir(DPO_MODELS_PATH)
for dpoRoute in dpoRoutes:
    model_path = os.path.join(DPO_MODELS_PATH, dpoRoute)
    model, tokenizer = cargar_modelo(model_path)
    responses = list(map(lambda prompt: generar_respuesta(model, tokenizer, prompt), prompts))
    result_df = pd.DataFrame({"prompt": prompts, "response": responses})
    result_df['algin']="DPO"
    result_df['model']=dpoRoute
    result_df.to_csv(os.path.join(RESPONSES_PATH, f"responses_DPO_{dpoRoute}.csv"), index=False)
    print(f"Respuestas generadas y guardadas para el modelo DPO: {dpoRoute}")
#respuestas PPO

ppoRoutes=os.listdir(PPO_MODELS_PATH)
for ppoRoute in ppoRoutes:
    model_path = os.path.join(PPO_MODELS_PATH, ppoRoute)
    model, tokenizer = cargar_modelo(model_path)
    responses = list(map(lambda prompt: generar_respuesta(model, tokenizer, prompt), prompts))
    result_df = pd.DataFrame({"prompt": prompts, "response": responses})
    result_df['algin']="PPO"
    result_df['model']=ppoRoute
    result_df.to_csv(os.path.join(RESPONSES_PATH, f"responses_PPO_{ppoRoute}.csv"), index=False)
    print(f"Respuestas generadas y guardadas para el modelo PPO: {ppoRoute}")