part_bash """
../bin/pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
"""
part_python """
from transformers import AutoTokenizer, AutoModelForCausalLM

device = \"mps\"
model_path = \"01-ai/Yi-Coder-9B-Chat\"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=\"auto\").eval()

prompt = \"Write a quick sort algorithm.\"
messages = [
    {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
    {\"role\": \"user\", \"content\": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors=\"pt\").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024,
    eos_token_id=tokenizer.eos_token_id
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
"""
