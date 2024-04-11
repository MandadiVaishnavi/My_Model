import numpy as np
import matplotlib.pyplot as plt  # Import the matplotlib library for visualization
import monai.transforms as mt
import torch
import nibabel as nib
from stl import mesh
from monai.data import (
    CacheDataset,                # A dataset that caches data in memory for faster access
    DataLoader,                 # DataLoader for iterating over batches of data
    Dataset,                    # Base dataset class for MONAI
    decollate_batch,             # Separate a batch of data into individual samples
  )
def predict(input_path):
    test_files = [
      {"image": input_path}
  ]
    device = torch.device("cpu")
    test_transform = mt.Compose([
        mt.ToTensorD(keys=["image"], allow_missing_keys=False)
        ])
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_data = DataLoader(test_loader_ACDC3(transform=test_transform, test_index=None), batch_size=1, shuffle=False)

