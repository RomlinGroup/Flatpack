part_bash """
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install cmake espeak-ng -y
"""
part_bash """
git clone --depth 1 --recursive https://github.com/RWKV/rwkv.cpp
"""
part_bash """
cp -f ../rwkv_cpp.py ./rwkv.cpp/python/rwkv_cpp.py
"""
part_bash """
cd rwkv.cpp
cmake .
cmake --build . --config Release

wget -nc https://huggingface.co/BlinkDL/rwkv-6-world/resolve/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth

../../bin/python python/convert_pytorch_to_ggml.py RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth RWKV-x060-World-1B6-v2.1-20240328-ctx4096.bin FP16

../../bin/python python/quantize.py RWKV-x060-World-1B6-v2.1-20240328-ctx4096.bin RWKV-x060-World-1B6-v2.1-20240328-ctx4096-Q4_0.bin Q4_0

../../bin/python python/rwkv_cpp.py \
RWKV-x060-World-1B6-v2.1-20240328-ctx4096-Q4_0.bin
"""