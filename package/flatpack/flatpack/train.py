import os
import time
import sys


def train(train_function, save_dir, model_type='rnn', framework='pytorch', *args, **kwargs):
    """
    A universal training function provided by flatpack.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Accessing command line arguments
    epochs = kwargs.get('epochs', 10)
    batch_size = kwargs.get('batch_size', 32)

    print(f"Training {model_type} model with epochs: {epochs} and batch size: {batch_size}")

    # Initialize a timer
    start_time = time.time()

    # Call the user-provided training function and update the progress after each epoch
    for epoch in range(epochs):
        output = train_function(*args, **kwargs)
        # Calculate and print the elapsed time
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Elapsed Time: {elapsed_time:.2f} seconds")

    # Calculate and display the total training time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    # Save the trained model based on the specified framework
    if framework == 'pytorch':
        import torch
        # Save the model with a filename based on the model_type
        torch.save(output['model'].state_dict(), os.path.join(save_dir, f'{model_type}_model.pth'))

    return output
