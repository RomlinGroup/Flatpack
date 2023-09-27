import json
import os
import time
import torch
from .datasets import prepare_text_dataset  # Import the prepare_text_dataset function
from .utils import configure_device  # Import the configure_device function


def build(url, user_train_function, save_dir, model_type='rnn', framework='pytorch', *args, **kwargs):
    os.makedirs(save_dir, exist_ok=True)

    # Use prepare_text_dataset to obtain dataset, char_to_index, and index_to_char
    dataset, char_to_index, index_to_char = prepare_text_dataset(url)

    # Use configure_device to obtain the device
    device = configure_device()

    epochs = kwargs.get('epochs', 10)
    batch_size = kwargs.get('batch_size', 32)

    print(f"🚀 Training {model_type} model with epochs: {epochs} and batch_size: {batch_size}")
    print(f"🖥 Model is set to train on {device}")  # Print device information here

    start_time = time.time()
    result = user_train_function(*args, **kwargs)
    model = result.get('model')

    elapsed_time = time.time() - start_time
    print(f"✅ Training completed in {elapsed_time:.2f} seconds")

    if framework == 'pytorch' and model is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, f'{model_type}_model.pth'))

    with open(os.path.join(save_dir, 'char_to_index.json'), 'w') as f:
        json.dump(char_to_index, f)

    with open(os.path.join(save_dir, 'index_to_char.json'), 'w') as f:
        json.dump(index_to_char, f)
