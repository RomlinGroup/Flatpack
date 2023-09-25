# flatpack/train.py

import os
import argparse


def train(train_function, save_dir, framework='pytorch', *args, **kwargs):
    """
    A universal training function provided by flatpack.

    :param train_function: The user-provided training function.
    :param save_dir: The directory where the model and checkpoints will be saved.
    :param framework: The framework being used (e.g., 'pytorch', 'tensorflow').
    :param args: Positional arguments to be passed to the train_function.
    :param kwargs: Keyword arguments to be passed to the train_function.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Accessing command line arguments
    epochs = kwargs.get('epochs', 10)  # Default value is 10 if not provided
    batch_size = kwargs.get('batch_size', 32)  # Default value is 32 if not provided

    print(f"Training model with epochs: {epochs} and batch size: {batch_size}")

    # Modify the train_function call if needed to pass epochs and batch_size
    output = train_function(*args, **kwargs)

    if framework == 'pytorch':
        import torch
        torch.save(output['model'].state_dict(), os.path.join(save_dir, 'model.pth'))

    return output
