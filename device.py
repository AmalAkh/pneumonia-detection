import torch
device_name = "mps" if torch.mps.is_available() else "cpu"

