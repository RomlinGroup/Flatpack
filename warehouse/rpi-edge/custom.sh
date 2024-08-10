part_bash """
git clone --depth 1 https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
"""
part_bash """
flatpack compress google/gemma-2-2b-it --token <hf_token>
"""
part_bash """
mv ./gemma-2-2b-it/gemma-2-2b-it-Q4_K_S.gguf \
./llama.cpp/models/gemma-2-2b-it-Q4_K_S.gguf
"""
part_bash """
cd llama.cpp

./llama-cli \
-m models/gemma-2-2b-it-Q4_K_S.gguf \
-p \"What is edge artificial intelligence? Provide a brief, one-sentence answer.\n---\n\" \
-n 64 \
> output.txt \
2>log.txt
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