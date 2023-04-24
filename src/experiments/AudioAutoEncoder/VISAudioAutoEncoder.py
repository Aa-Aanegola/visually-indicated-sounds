import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from transformers import VideoMAEForVideoClassification
from typing import Dict, Any

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
            nn.Conv1d(1, 16, 512, stride=16, padding=8, padding_mode='replicate'),
            nn.BatchNorm1d(16),
            nn.RReLU(), 
            nn.Conv1d(16, 64, 256, stride=4, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(64), 
            nn.RReLU(), 
            nn.Conv1d(64, 8, 64, stride=2, padding=1, padding_mode='replicate'),
            nn.BatchNorm1d(8))        
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 64, 64, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.RReLU(), 
            nn.ConvTranspose1d(64, 16, 256, stride=4, padding=1),
            nn.BatchNorm1d(16),
            nn.RReLU(), 
            nn.ConvTranspose1d(16, 1, 512, stride=16, padding=8))
        
        self.loss_fn = nn.MSELoss()
        
        
    def forward(self, wav):
        emb = self.encoder(wav.unsqueeze(1))
        reconstructed = self.decoder(emb).squeeze()
        return reconstructed
        
    def get_encoding(self, wav):
        return self.encoder(wav.unsqueeze(1)).flatten(start_dim=1)

    def get_reconstructed(self, emb):
        return self.decoder(emb.reshape(-1, 8, 310)).squeeze()
    
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

class VISVMAEModel(pl.LightningModule):
    def __init__(self, aae):
        super().__init__()
        self.featureExtractor = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
        self.featureExtractor.classifier = nn.Identity()
        

        for param in self.featureExtractor.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2480)
        )
        
        self.aae = aae
        for param in self.aae.parameters():
            param.requires_grad = False

    def forward(self, X):
        X = self.featureExtractor(X).logits
        X.detach_()

        out = self.mlp(X)
        return out
    
    def _common_step(self, batch, batch_idx):
        frames, material, wav = batch
        out = self(frames)
        audio_emb = self.aae.get_encoding(wav)
        loss = nn.MSELoss()(out, audio_emb)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-4)
    
    def get_waveform(self, frames):
        out = self(frames)
        return self.aae.decoder(out)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)

        modified_state_dict = {}
        for k, v in self.mlp.state_dict().items():
            modified_state_dict[f'mlp.{k}'] = v

        checkpoint['state_dict'] = modified_state_dict

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        
        modified_state_dict = {}
        for k, v in self.featureExtractor.state_dict().items():
            modified_state_dict[f'featureExtractor.{k}'] = v
            
        checkpoint['state_dict'].update(modified_state_dict)
        return super().on_load_checkpoint(checkpoint)        
