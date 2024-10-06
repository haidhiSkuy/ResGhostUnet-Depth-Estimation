import torch
import lightning as L 
from model.residual_ghost_net import ResidualGhostUNet 
from torchmetrics.image import StructuralSimilarityIndexMeasure 


class Losses: 
    def __init__(self) -> None:
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0) 
    
    def ssim_loss(self, preds, targets): 
        return 1 - self.ssim(preds, targets)

class ResUnetLightning(L.LightningModule): 
    def __init__(self, learning_rate : float = 1e-5): 
        super().__init__()
        self.learning_rate = learning_rate 
        self.model = ResidualGhostUNet()  

        self.losses = Losses()

        self.save_hyperparameters()     
        
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)

        y = y.unsqueeze(1)

        loss = self.losses.ssim_loss(prediction, y)
        self.log("ssim_train_loss", loss, on_epoch=True, on_step=False)
        return loss 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)

        y = y.unsqueeze(1)

        loss = self.losses.ssim_loss(prediction, y)
        self.log("ssim_val_loss", loss, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    
