# 🐳 Use the nvidia/cuda:12.1.1-base-ubuntu22.04 image as the base image
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# 🏷️ Set the version label of the image to 0.8
LABEL version="0.8"

# 📁 Create a directory called flatpack-ai in the /home directory
RUN mkdir /home/flatpack-ai

# 📦 Update the package list and upgrade all installed packages
# Then, install build-essential, git, and python3 packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y build-essential git python3

# 🐾 Set the default command to tail the /dev/null file
# 🛑 This is done to prevent the container from exiting immediately
CMD tail -f /dev/null