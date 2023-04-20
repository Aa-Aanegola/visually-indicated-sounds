import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torchvision.models import resnet18, ResNet18_Weights

class VISBiLSTMModel(pl.LightningModule):

    def __init__(self, outputSize: int=42, isBi: bool=False, hidden_dim: int=256):
        super().__init__()
        self.featureExtractor = resnet18(ResNet18_Weights.DEFAULT)
        self.featureExtractor.fc = nn.Identity()

        for param in self.featureExtractor.parameters():
            param.requires_grad = False

        fc_hidden_dim = 2*hidden_dim if isBi else hidden_dim
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True, bidirectional=isBi)
        self.fc = nn.Linear(fc_hidden_dim, outputSize)

        self.loss_fn = nn.MSELoss()

    def forward(self, stFrames, frame0):

        # stFrames: batchx45x224x224x3
        # frame0: batchx224x224x3

        stFrameFeatures = []
        for i in range(stFrames.shape[1]):
            currStFrame = stFrames[:,i,:,:,:].squeeze(1)
            currStFrameFeatures = self.featureExtractor(currStFrame)
            stFrameFeatures.append(currStFrameFeatures)
        stFrameFeatures = torch.stack(stFrameFeatures, dim=1)

        # tried but didn't do well
        # frame0Features = self.featureExtractor(frame0).unsqueeze(1).repeat(1, stFrames.shape[1], 1)
        # X = torch.cat([stFrameFeatures, frame0Features], dim=2)
        X = stFrameFeatures
        
        X.detach_()
        # X is the input to the LSTM -> batchx45x1024
        X, _ = self.lstm(X)

        out = self.fc(X).transpose(1, 2)
        
        return out
    
    def training_step(self, batch, batch_idx):
        coch, stFrames, frame0, material = batch
        out = self(stFrames, frame0)
        loss = self.loss_fn(out, coch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        coch, stFrames, frame0, material = batch
        out = self(stFrames, frame0)
        loss = self.loss_fn(out, coch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer