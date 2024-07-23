disabled part_bash """
echo \"Hello, World!\"
"""
part_bash """
./llama-cli \
-m models/gemma-1.1-2b-it-Q4_K_M.gguf \
-p \"What is the meaning of life?\" \
-n 64 \
> output.txt \
2>/dev/null
"""
part_bash """
# Check output.txt

if [ -f \"output.txt\" ]; then
    echo \"🪝 Running hooks for output.txt:\"
    echo \"----------------------\"
    cat output.txt
else
    echo \"Error: output.txt does not exist.\"
fi
"""