import pandas  as pd
import os
import torch 
from pathlib import Path
from evaluateResponsesTools import evaluateResponses

# ============================================================
# 1. Configuración
# ============================================================

ROUTE_BERT="/workspace/models/deberta-v3-large"
ROUTE_CABEZALES="/workspace/adaptedModels/cabezales"
ROUTE_ENCODER="/workspace/papaerRLAIF/codTraining/cabezales/encoders"
ROUTE_RESPONSES="/workspace/papaerRLAIF/makeResponses/responses" 
OUTPUT_ROUTE="/worksapce/papaerRLAIF/makeResponses/evaluatedResponses"
MAX_LENGTH = 128
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Dispositivo utilizado: {DEVICE}")


# ============================================================
# 2. calificamos por modelo  y por cabezal
# ============================================================

cabezales=os.listdir(ROUTE_CABEZALES)
bases=os.listdir(ROUTE_RESPONSES)
encoders=os.listdir(ROUTE_ENCODER)


for base in bases:
    data=pd.read_csv(f"{ROUTE_RESPONSES}/{base}")
    print(f"{base} CARGADA",data.shape[0])
    resultBase=data['response'].copy()
    for i in range(len(cabezales)):
        cabezal=cabezales[i]
        cabezalName=cabezal.split(".")[0]
        encoder=encoders[i]
        if cabezal.split(".")[0] != encoder.split(".")[0]:
            raise ValueError("no  ciciden  el  cabezal  y elencoder")
        
        evaluated=evaluateResponses(f"{ROUTE_RESPONSES}/{base}",'response',
                        ROUTE_BERT,
                        Path(f"{ROUTE_CABEZALES}/{cabezal}"),
                        Path(f"{ROUTE_ENCODER}/{encoder}"),BATCH_SIZE,MAX_LENGTH,
                        DEVICE)
        columns=[f"{x}_{cabezalName}" if x != 'response' else x for  x in evaluated.columns ]
        evaluated.columns=columns
        print("RESULTADOS EVALUADOS")
        print(evaluated.columns)
        
        if evaluated.shape[0] != resultBase.shape[0]:
            raise ValueError(' shape  de resultados  inconsistente')
        
        
        resultBase=pd.merge(
            resultBase, evaluated,on='response', how='inner'
        )
        print("RESULTADOS CONCATENADOS",resultBase.shape)
    
    resultBase.to_csv(f"{OUTPUT_ROUTE}/evaluated_{base}")
    print(f"{base} TOTALMENTE EVALUADA")


        

