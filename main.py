import torch 
import wandb
import numpy as np 
import lightning as L
from model.lightning_model import ResUnetLightning
from lightning.pytorch.loggers import WandbLogger
from data.lightning_dataset import DepthDataModule
from lightning.pytorch.callbacks import ModelCheckpoint  

# Callback config
checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoint',
    filename='resghost-{epoch:02d}-{val_loss:.2f}', 
    save_top_k=1, 
    save_on_train_epoch_end=True,
    verbose=True, 
    save_weights_only=True
)

# Wandb Config
wandb.login(key="06ee7ca7307838ddb249c4cda6662d79e7d7d16d")  
wandb_logger = WandbLogger(project="Jurnal_ResGhostUnet_Depth", log_model="all")   

# Dataset
data = DepthDataModule(
    batch_size=16,  
    train_dataset_path="train_dataset.pth", 
    val_dataset_path="val_dataset.pth"
    )

# Model
model = ResUnetLightning(learning_rate=0.0005) 
trainer = L.Trainer(
    max_epochs=50, 
    logger=wandb_logger, 
    callbacks=[checkpoint],
    devices=-1
) 

# Training 
trainer.fit(model, data)
wandb.finish()