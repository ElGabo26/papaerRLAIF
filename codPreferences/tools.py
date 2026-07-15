from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def  dowloadModel(modelroot:str, localRoot:str):
    nameModel=modelroot.split("/")[-1]
    localRoot=localRoot+"/"+nameModel
    ruta_modelo = snapshot_download(
    repo_id=modelroot,
    local_dir=localRoot)
    print("Modelo descargado en:", ruta_modelo)
    return ruta_modelo



def testModel(rutaModelo):
    #carga  del  modelo
    tokenizer = AutoTokenizer.from_pretrained(
        rutaModelo,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        rutaModelo,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True
    )

    print("Modelo cargado correctamente")
    print("Arquitectura:", model.__class__.__name__)
    
    #prueba de funcionamiento
    mensajes=[{
        "role": "user",
        "content":"make a  question for elementary  school student  to learn addition"
    }]
    
    texto=tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_promot=True
    )
    
    inputs=tokenizer(
        texto,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        salida = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id )

    respuesta = tokenizer.decode(
        salida[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True)

    print(respuesta)
    return tokenizer, model

def makeResponse(tokenizer, model, prompt, temperature=0.3):
    mensajes=[{
        "role": "user",
        "content":prompt
    }]
    #aplicamos el  formatopara el modelo en  especifico
    texto=tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_promot=True
    )
    #transformamos  el  texto a  tokens
    inputs=tokenizer(
        texto,
        return_tensors="pt"
    ).to(model.device)
    #generamos la  salida  del  modelo   este  retorna  el  imput+ generated
    with torch.no_grad():
        salida = model.generate(
        **inputs, #se  agrega  todos  los  inputs
        max_new_tokens=200,
        do_sample=True, # Garantiza aleatoriedad
        temperature=temperature, #Temperatura
        top_p=0.9, # top  de  probabilidad  acumulada
        pad_token_id=tokenizer.eos_token_id )
#obtenemos solo  la  respuesta  generada
    respuesta = tokenizer.decode(
        salida[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True)
    
    return respuesta

    
    
