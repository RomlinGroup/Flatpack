part_bash """
sudo apt update
sudo apt install git g++ wget build-essential

curl -L -o piper_arm64.tar.gz https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz

tar -xzf piper_arm64.tar.gz

curl -L -o piper/en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx

curl -L -o piper/en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

git clone https://github.com/ggerganov/llama.cpp

cd llama.cpp
make -j

curl -L -o models/Phi-3.1-mini-4k-instruct-Q4_K_S.gguf https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF/resolve/main/Phi-3.1-mini-4k-instruct-Q4_K_S.gguf
"""
disabled part_bash """
./llama-cli \
-m models/Phi-3.1-mini-4k-instruct-Q4_K_S.gguf \
-p \"What is edge artificial intelligence?\" \
-n 400 \
> output.txt \
2>/dev/null
"""
disabled part_bash """
if [ -f \"output.txt\" ]; then
    echo \"Running hooks for output.txt:\"
    cat output.txt
else
    echo \"Error: output.txt does not exist.\"
fi
"""