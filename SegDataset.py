from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class SegDataset(Dataset):
    def __init__(self, file_pairs, transform=None):
        self.file_pairs = file_pairs
        self.transform = transform

    def __getitem__(self, idx):
        img_path, mask_path = self.file_pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)

        # 마스크 이진화
        mask = np.array(mask)
        mask = (mask == 2).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return image, mask

    def __len__(self):
        return len(self.file_pairs)