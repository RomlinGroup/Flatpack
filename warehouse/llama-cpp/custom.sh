# Check if the build was already done by looking for a build marker file
if [ -f ".build_successful" ]; then
    echo "✅ Build already completed for llama.cpp"
else
    echo "🦙 Building llama.cpp"
    # Run make and check if it succeeded
    if make; then
        echo "✅ Finished building llama.cpp"
        # Create a marker file to indicate the build was successful
        touch .build_successful
    else
        echo "❌ Build failed for llama.cpp"
        exit 1
    fi
fi