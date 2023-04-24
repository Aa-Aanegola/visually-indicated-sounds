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
        
        
# class AudioAutoEncoderConv(pl.LightningModule):
#     def __init__(self, input_size=12000):
#         super().__init__()
#         self.encoder = {
#             'conv1':nn.Conv1d(1, 4, 512), 
#             'pool1':nn.MaxPool1d(32),
#             'relu1':nn.ReLU(), 
#             'conv2':nn.Conv1d(4, 8, 256), 
#             'pool2':nn.MaxPool1d(16), 
#             'relu2':nn.ReLU(), 
#             'conv3':nn.Conv1d(8, 16, 128)   
#         }
        
#         self.decoder = {
#             'deconv1':nn.ConvTranspose1d(16, 8, 128),
#             'relu1':nn.ReLU(), 
#             'unpool1':nn.MaxUnpool1d(16),  
#             'deconv2':nn.ConvTranspose1d(8, 4, 256), 
#             'relu2':nn.ReLU(), 
#             'unpool2':nn.MaxUnpool1d(32), 
#             'deconv3':nn.ConvTranspose1d(4, 1, 512)}
        
        
#     def forward(self, wav):
#         # Encode
#         x, indices1 = self.encoder['pool1'](self.encoder['conv1'](wav.unsqueeze(1)))
#         x, indices2 = self.encoder['conv2'](self.encoder['relu1'](x))
        
#     def get_encoding(self, wav):
#         return self.encoder(wav)

#     def get_reconstructed(self, emb):
#         return self.decoder(emb)
    
#     def training_step(self, batch, batch_idx):
#         out = self(batch)
#         loss = nn.MSELoss()(out, batch)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         out = self(batch)
#         loss = nn.MSELoss()(out, batch)
#         self.log('val_loss',  loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
    
#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), 1e-4)
#         return optimizer 