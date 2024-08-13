part_bash """
../bin/pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
"""
part_bash """
../bin/pip install rwkv
"""
part_bash """
# https://huggingface.co/BlinkDL/rwkv-6-world (Apache-2.0)

wget -nc https://huggingface.co/BlinkDL/rwkv-6-world/resolve/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth
"""
part_python """
import os

os.environ['RWKV_JIT_ON'] = '1'
os.environ[\"RWKV_CUDA_ON\"] = '0'

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

model = RWKV(
    model='RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
    strategy='cpu fp32'
)

pipeline = PIPELINE(model, \"rwkv_vocab_v20230424\")

prompt = \"What is edge artificial intelligence?\"
ctx = f\"System: You are a helpful assistant. Please respond in one short sentence.\nUser: {prompt}\nAssistant:\"

args = PIPELINE_ARGS(
    alpha_decay=0.996,
    alpha_frequency=0.25,
    alpha_presence=0.25,
    chunk_len=64,
    temperature=0.7,
    token_ban=[0],
    token_stop=[],
    top_k=50,
    top_p=0.5
)

output = \"\"
stop_token = \". \"
unexpected_tokens = [\"Assistant:\", \"User:\"]

def generate_until_stop(ctx, stop_token):
    global output
    def controlled_print(s):
        global output
        output += s

        if stop_token in output or any(token in output for token in unexpected_tokens):
            raise StopIteration
    try:
        pipeline.generate(
            args=args,
            callback=controlled_print,
            ctx=ctx,
            token_count=64
        )
    except StopIteration:
        pass

generate_until_stop(ctx, stop_token)

cleaned_output = output.split(stop_token)[0].replace(\"\n\", \" \").strip() + '.'

for token in unexpected_tokens:
    if token in cleaned_output:
        cleaned_output = cleaned_output.split(token)[0].strip() + '.'

with open(\"output.txt\", \"w\") as file:
    file.write(cleaned_output)
"""
part_bash """
if [ ! -d \"piper\" ]; then
    wget -nc https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
    tar -xvzf piper_arm64.tar.gz
else
    echo \"The piper folder already exists. No actions were taken.\"
fi

# LJSpeech (medium)
# License: public domain
# https://brycebeattie.com/files/tts/

wget -nc https://sfo3.digitaloceanspaces.com/bkmdls/lj-med.onnx
wget -nc https://sfo3.digitaloceanspaces.com/bkmdls/lj-med.onnx.json
"""
part_python """
import os
import subprocess
import tempfile

from multiprocessing import Pool

def process_chunk(chunk, model_file, piper_path):
	with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
		command = [piper_path, \"--model\", model_file, \"--output_file\", temp_file.name]
		try:
			result = subprocess.run(command, input=chunk, text=True, check=True, capture_output=True)
			if result.returncode == 0:
				print(f\"Processed chunk of {len(chunk)} characters successfully.\")
				return temp_file.name
			else:
				print(f\"Error processing chunk: {result.stderr}\")
				return None
		except subprocess.CalledProcessError as e:
			print(f\"Error occurred processing chunk: {e}\")
			return None

def combine_audio_files(temp_files, final_output_file):
	combine_command = [\"sox\"] + temp_files + [final_output_file]
	try:
		subprocess.run(combine_command, check=True)
		print(f\"All chunks combined successfully. Output saved to: {final_output_file}\")
	except subprocess.CalledProcessError as e:
		print(f\"Error occurred while combining audio files: {e}\")

def process_and_combine(chunks, model_file, piper_path, final_output_file):
	temp_files = []
	with Pool(processes=4) as pool:
		temp_files = pool.starmap(process_chunk, [(chunk, model_file, piper_path) for chunk in chunks])

	temp_files = [f for f in temp_files if f]
	if temp_files:
		combine_audio_files(temp_files, final_output_file)

		for temp_file in temp_files:
			os.remove(temp_file)

final_output_file = \"speech.wav\"
input_file = \"output.txt\"
model_file = \"lj-med.onnx\"
piper_path = os.path.abspath(\"./piper/piper\")

with open(input_file, 'r') as file:
	input_text = file.read()

print(f\"Total text length: {len(input_text)} characters\")

chunks = []
current_chunk = \"\"
chunk_size = 50
words = input_text.split()

for word in words:
	if len(current_chunk) + len(word) + 1 <= chunk_size:
		if current_chunk:
			current_chunk += \" \" + word
		else:
			current_chunk = word
	else:
		chunks.append(current_chunk)
		current_chunk = word

if current_chunk:
	chunks.append(current_chunk)

process_and_combine(chunks, model_file, piper_path, final_output_file)

print(f\"All chunks processed and saved to: {final_output_file}\")
"""