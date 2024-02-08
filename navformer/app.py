import os
import time
import torch

try:
    import tiktoken
except ImportError:
    raise ImportError("Please ensure the 'tiktoken' library is installed by running `pip install tiktoken`.")

try:
    from nanoGPT.model import GPTConfig, GPT
except ImportError as e:
    if str(e) == "No module named 'nanoGPT'":
        raise ImportError("The 'nanoGPT' repository is not found.")
    else:
        raise ImportError("Please ensure 'model.py' is available.")

# Configuration settings
compile_model = True
device = 'cpu'
dtype = 'bfloat16'
max_new_tokens = 128
temperature = 0.8
top_k = 5

# Enabling TensorFlow 32-bit computation for CUDA operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Determining device type based on the 'device' variable
device_type = 'cuda' if 'cuda' in device else 'cpu'
# Mapping dtype strings to torch.dtype objects
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# Model checkpoint loading
ckpt_path = os.path.join('navformer.pt')
try:
    checkpoint = torch.load(ckpt_path, map_location=device)
except FileNotFoundError:
    raise FileNotFoundError(f"Checkpoint file not found in {ckpt_path}. Please check the path.")

# GPT model configuration and initialization
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'

# Removing unwanted prefix from model keys
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

if compile_model:
    model = torch.compile(model)

# Loading GPT-2 encodings
print("Loading GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")

# Encoding and decoding helpers
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: enc.decode(l)


# Function to respond to input text
def respond(input_text, max_new_tokens, temperature, top_k):
    x = torch.tensor(encode(input_text), dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type, dtype=ptdtype):
            start_time = time.time()
            generated = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference Time: {inference_time:.2f} seconds")
            output = decode(generated[0].tolist())
            return output


# Main execution block
if __name__ == "__main__":
    while True:  # Start an infinite loop
        user_input = input("Enter your command (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':  # Check if the user wants to exit
            print("Exiting...")
            break  # Exit the loop and end the program

        start = '<command> ' + user_input + '<endOfText>'
        full_input = start

        response = respond(full_input, max_new_tokens, temperature, top_k)

        response_parts = response.split('<command>')
        first_response = '<command>' + response_parts[1] if len(response_parts) > 1 else response

        print(first_response)
