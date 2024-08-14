part_bash """
nvidia-smi
"""
part_bash """
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb

sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update

sudo apt-get -y install cuda-toolkit-12-6
sudo apt-get -y install nvidia-cuda-toolkit
"""
part_bash """
nvcc --version
"""