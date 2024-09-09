part_bash """
if [ ! -f mmproj-model-f16.gguf ]; then
    wget -nc -O mmproj-model-f16.gguf \"https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/resolve/main/mmproj-model-f16.gguf\"
else
    echo \"Model already exist.\"
fi
"""
part_bash """
if [ ! -f ggml-model-Q4_K_M.gguf ]; then
    wget -nc -O ggml-model-Q4_K_M.gguf \"https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/resolve/main/ggml-model-Q4_K_M.gguf\"
else
    echo \"Model already exist.\"
fi
"""
part_bash """
if [ -f ../tiger.png ]; then
    cp -f ../tiger.png tiger.png
fi
"""
part_bash """
./llama-minicpmv-cli \
--image tiger.png \
--mmproj mmproj-model-f16.gguf \
--repeat-penalty 1.05 \
--temp 0.7 \
--top-k 100 \
--top-p 0.8 \
-c 4096 \
-m ggml-model-Q4_K_M.gguf \
-p \"Should I pet this dog?\" \
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
