disabled part_bash """
wget https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/resolve/main/mmproj-model-f16.gguf
"""
disabled part_bash """
wget https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/resolve/main/ggml-model-Q4_K_M.gguf
"""
part_bash """
./minicpmv-cli \
-m ggml-model-Q4_K_M.gguf \
--mmproj mmproj-model-f16.gguf \
-c 4096 \
--temp 0.7 \
--top-p 0.8 \
--top-k 100 \
--repeat-penalty 1.05 \
--image tiger.png \
-p \"Should I pet this dog?\" \
> output.txt \
2>/dev/null
"""
part_bash """
if [ -f \"output.txt\" ]; then
    echo \"🪝 Running hooks for output.txt:\"
    cat output.txt
else
    echo \"Error: output.txt does not exist.\"
fi
"""