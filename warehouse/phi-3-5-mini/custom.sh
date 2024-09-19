part_bash """
if [ ! -f models/Phi-3.5-mini-instruct-Q4_K_S.gguf ]; then
    wget -nc -O models/Phi-3.5-mini-instruct-Q4_K_S.gguf \"https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_S.gguf\"
else
    echo \"Model already exist.\"
fi
"""
part_bash """
./llama-cli \
-m models/Phi-3.5-mini-instruct-Q4_K_S.gguf \
-n 64 \
-p \"What is the meaning of life?\" \
> output.txt \
2>log.txt
"""
part_bash """
if [ -f \"output.txt\" ]; then
    cat output.txt
else
    echo \"Error: output.txt does not exist.\"
fi
"""
