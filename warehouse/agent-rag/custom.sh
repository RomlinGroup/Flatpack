part_bash """
wget -nc -q https://raw.githubusercontent.com/RomlinGroup/Flatpack/main/agents/fast_api_local.py

wget -nc -q https://huggingface.co/ggml-org/gemma-1.1-2b-it-Q4_K_M-GGUF/resolve/main/gemma-1.1-2b-it.Q4_K_M.gguf
"""
part_bash """
flatpack agents spawn fast_api_local.py
"""
