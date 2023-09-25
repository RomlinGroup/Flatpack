import os
from tqdm import tqdm


def train(train_function, save_dir, model_type='rnn', framework='pytorch', *args, **kwargs):
    """
    A universal training function provided by flatpack.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Accessing command line arguments
    epochs = kwargs.get('epochs', 10)
    batch_size = kwargs.get('batch_size', 32)

    print(f"Training {model_type} model with epochs: {epochs} and batch size: {batch_size}")

    # Initialize a progress bar using tqdm
    with tqdm(total=epochs, desc='Training', unit='epoch') as pbar:
        # Call the user-provided training function and update the progress bar after each epoch
        for epoch in range(epochs):
            output = train_function(*args, **kwargs)
            # Update the progress bar and display the loss
            pbar.set_postfix(loss=output.get('loss', 'N/A'))
            pbar.update(1)

    # Save the trained model based on the specified framework
    if framework == 'pytorch':
        import torch
        # Save the model with a filename based on the model_type
        torch.save(output['model'].state_dict(), os.path.join(save_dir, f'{model_type}_model.pth'))

    return output
