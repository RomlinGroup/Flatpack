import random
import torch


def configure_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_random(seed_value):
    random.seed(seed_value)
