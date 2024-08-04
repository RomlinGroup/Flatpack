disabled part_bash """
sudo apt update
sudo apt install git g++ wget build-essential

curl -L -o piper_arm64.tar.gz https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz

tar -xzf piper_arm64.tar.gz

curl -L -o piper/en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx

curl -L -o piper/en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

git clone https://github.com/ggerganov/llama.cpp

cd llama.cpp
make -j
"""
disabled part_bash """
flatpack compress google/gemma-2-2b-it --token <hf_token>
"""
part_bash """
cd llama.cpp

./llama-cli \
-m models/gemma-2-2b-it-Q4_K_S.gguf \
-p \"What is edge artificial intelligence?\" \
-n 128 \
> output.txt \
2>/dev/null
"""
part_bash """
if [ -f \"output.txt\" ]; then
    echo \"Running hooks for output.txt:\"
    cat output.txt
else
    echo \"Error: output.txt does not exist.\"
fi
"""