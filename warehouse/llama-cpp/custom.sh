# "${VENV_PIP}"
# "${VENV_PYTHON}"

if [ -f ".build_successful" ]; then
    echo "✅ Build already completed for llama.cpp"
else
    echo "🦙 Building llama.cpp"
    if make; then
        echo "✅ Finished building llama.cpp"
        touch .build_successful
    else
        echo "❌ Build failed for llama.cpp"
        exit 1
    fi
fi