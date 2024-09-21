part_bash """
../bin/pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
"""
part_python """
from phi_3_vision_mlx import train_lora

train_lora(
    lora_layers=5,
    lora_rank=16,
    epochs=10,
    lr=1e-4,
    warmup=0.5,
    dataset_path=\"JosefAlbers/akemiH_MedQA_Reason\"
)
"""
part_python """
from phi_3_vision_mlx import generate, test_lora

generate(
    \"Describe the potential applications of CRISPR gene editing in medicine.\",
    blind_model=True,
    quantize_model=True,

    use_adapter=True
)

test_lora()
"""
