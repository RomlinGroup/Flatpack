part_bash """
../bin/pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
"""
part_python """
import scipy

from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained(\"suno/bark\")

model = BarkModel.from_pretrained(\"suno/bark-small\")

inputs = processor(\"\"\" What is the meaning of life? \"\"\")

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

sample_rate = model.generation_config.sample_rate

scipy.io.wavfile.write(\"output.wav\",rate=sample_rate, data=audio_array)
"""
