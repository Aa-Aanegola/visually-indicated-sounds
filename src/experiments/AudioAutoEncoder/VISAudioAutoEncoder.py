import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

class AudioAutoEncoder(pl.LightningModule):
    def __init__(self, input_size=12000):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(), 
            nn.Linear(input_size//2, input_size//4), 
            nn.ReLU(), 
            nn.Linear(input_size//4, 512))
        
        self.decoder = nn.Sequential(
            nn.Linear(512, input_size//4), 
            nn.ReLU(), 
            nn.Linear(input_size//4, input_size//2), 
            nn.ReLU(), 
            nn.Linear(input_size//2, input_size))
        
    def forward(self, wav):
        encoding = self.encoder(wav)
        reconstructed = self.decoder(encoding)
        return reconstructed
        
    def get_encoding(self, wav):
        return self.encoder(wav)

    def get_reconstructed(self, emb):
        return self.decoder(emb)
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = nn.MSELoss()(out, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = nn.MSELoss()(out, batch)
        self.log('val_loss',  loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 1e-4)        
        
        
class AudioAutoEncoderConv(pl.LightningModule):
    def __init__(self, input_size=48000):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 512, stride=16, padding=8, padding_mode='replicate'),
            nn.RReLU(), 
            nn.Conv1d(8, 16, 256, stride=4, padding=2, padding_mode='replicate'),
            nn.RReLU(), 
            nn.Conv1d(16, 32, 32, stride=2, padding=1, padding_mode='replicate'))        
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 32, stride=2, padding=1),
            nn.RReLU(), 
            nn.ConvTranspose1d(16, 8, 256, stride=4, padding=1),
            nn.RReLU(), 
            nn.ConvTranspose1d(8, 1, 512, stride=16, padding=8))
        
        self.loss_fn = nn.MSELoss()
        
        
    def forward(self, wav):
        emb = self.encoder(wav.unsqueeze(1))
        reconstructed = self.decoder(emb).squeeze()
        return reconstructed
        
    def get_encoding(self, wav):
        return self.encoder(wav.unsqueeze(1))

    def get_reconstructed(self, emb):
        return self.decoder(emb).squeeze()
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss_fn(out, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss_fn(out, batch)
        self.log('val_loss',  loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 1e-3)
        return optimizer 