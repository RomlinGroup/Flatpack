part_bash """
../bin/pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
"""
part_bash """
../bin/pip install accelerate datamodel_code_generator jsonschema
"""
part_python """
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

path = \"openbmb/MiniCPM3-4B\"
device = \"cpu\"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

messages = [
    {\"role\": \"user\", \"content\": \"What is the meaning of life?\"},
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors=\"pt\", add_generation_prompt=True).to(device)

model_outputs = model.generate(
    model_inputs,
    max_new_tokens=1024,
    top_p=0.7,
    temperature=0.7
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)
"""
