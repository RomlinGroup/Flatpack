part_bash """
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j
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
-p \"What is edge artificial intelligence?\" \
-n 256 \
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