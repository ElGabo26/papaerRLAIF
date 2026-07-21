import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


ADAPTER_PATH = input("modify  model  route: ")

use_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
)

dtype = (
    torch.bfloat16
    if use_bf16
    else torch.float16
)


tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_PATH,
    torch_dtype=dtype,
    device_map="auto",
)

model.eval()


messages = [
    {
        "role": "user",
        "content": (
            "Genera una pregunta clara de suma para estudiantes "
            "de segundo grado. Responde únicamente con la pregunta."
        ),
    }
]


input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
)

device = next(model.parameters()).device
input_ids = input_ids.to(device)


with torch.inference_mode():
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


generated_tokens = output_ids[0, input_ids.shape[-1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print(response)