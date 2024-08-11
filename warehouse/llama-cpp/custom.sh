part_bash """
if [ ! -f models/gemma-1.1-2b-it.Q4_K_M.gguf ]; then
    wget -O models/gemma-1.1-2b-it.Q4_K_M.gguf \"https://huggingface.co/ggml-org/gemma-1.1-2b-it-Q4_K_M-GGUF/resolve/main/gemma-1.1-2b-it.Q4_K_M.gguf\"
else
    echo \"Model already exist.\"
fi
"""
part_bash """
./llama-cli \
-m models/gemma-1.1-2b-it.Q4_K_M.gguf \
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