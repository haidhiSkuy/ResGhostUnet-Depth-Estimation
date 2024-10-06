import torch
import lightning as L 
from typing import List
from torch.utils.data import DataLoader
from data.torch_dataset import DepthDataset
from sklearn.model_selection import train_test_split

class DepthDataModule(L.LightningDataModule): 
    def __init__(
            self, 
            batch_size : int = 16,  
            train_size : float = 0.8,
            image_tensor : List[torch.tensor] = None, 
            label_tensor : List[torch.tensor] = None
        ) -> None:
        super().__init__()
        self.batch_size = batch_size 
        self.train_size = train_size
        self.image_tensor = image_tensor
        self.label_tensor = label_tensor

    def setup(self, stage: str): 
        X_train, X_test, y_train, y_test = train_test_split(
            self.image_tensor, self.label_tensor, 
            test_size=self.train_size, random_state=42
        )
        self.train_dataset = DepthDataset(X_train, y_train)
        self.val_dataset = DepthDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)