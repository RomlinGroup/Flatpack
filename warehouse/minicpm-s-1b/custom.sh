part_bash """
if [ ! -f models/MiniCPM-S-1B-sft.gguf ]; then
    wget -nc -q -O models/MiniCPM-S-1B-sft.gguf \"https://huggingface.co/openbmb/MiniCPM-S-1B-sft-gguf/resolve/main/MiniCPM-S-1B-sft.gguf\"
else
    echo \"Model already exist.\"
fi
"""
part_bash """
./build/bin/main -m models/MiniCPM-S-1B-sft.gguf -n 64 -t 1 -p \"What is the meaning of life?\"
"""
