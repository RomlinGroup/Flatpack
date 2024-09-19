part_bash """
../bin/pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
"""
part_bash """
generate(\'What is shown in this image?\', \'https://collectionapi.metmuseum.org/api/collection/v1/iiif/344291/725918/main-image\')
"""
