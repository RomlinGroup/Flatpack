# Use an official Ubuntu as a base image
FROM ubuntu:24.04

# Set the working directory in the container
WORKDIR /app

# Install necessary packages including Python, pip, and venv
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv venv

# Activate the virtual environment and install flatpack
RUN . venv/bin/activate && pip install flatpack

# Verify the installation by checking the flatpack version
RUN . venv/bin/activate

# Copy the script into the container
COPY custom.sh .

# Make the hello_world.sh script executable
RUN chmod +x custom.sh

# Run the hello_world.sh script within the virtual environment
CMD ["sh", "-c", ". venv/bin/activate && ./custom.sh"]