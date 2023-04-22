
import sys

sys.path.append('../..')

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
import glob
import os
import pickle
from tqdm import tqdm
from scipy.signal import resample

torch.set_float32_matmul_precision('medium')


class AudioAutoEncoder(pl.LightningModule):
    def __init__(self, input_size=12000):
        super().__init__()
        self.bottle_neck_dim = 256

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2*input_size//3),
            nn.ReLU(), 
            nn.Linear(2*input_size//3, input_size//3), 
            nn.ReLU(), 
            nn.Linear(input_size//3, self.bottle_neck_dim), 
            nn.ReLU())
        
        self.decoder = nn.Sequential(
            nn.Linear(self.bottle_neck_dim, input_size//3), 
            nn.ReLU(), 
            nn.Linear(input_size//3, 2*input_size//3), 
            nn.ReLU(), 
            nn.Linear(2*input_size//3, input_size))
        
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
        optimizer = optim.AdamW(self.parameters(), 1e-3)
        return optimizer 

class AudioDataset(Dataset):
    def __init__(self, root: str, sr: int=12000):
        self.root = root
        self.files = glob.glob(os.path.join(self.root, '*.pkl'))
        self.sr = sr

        self.wavs = []
        for fileName in tqdm(self.files, desc='Loading files into RAM'):
            with open(fileName, 'rb') as f:
                wav = pickle.load(f)
                if wav.shape[0]:
                    self.wavs.append(wav)
        
    def __len__(self):
        return len(self.wavs)
    
    def __getitem__(self, idx):
        wav = self.wavs[idx]
        downsampled = resample(wav, self.sr)
        downsampled = downsampled / np.max(np.abs(downsampled))

        return torch.tensor(downsampled, dtype=torch.float32)
    
if __name__=='__main__':
    BATCH_SIZE = 16
    NUM_WORKERS = 15
    EPOCHS = 100

    trainDataset = AudioDataset('../../../../audio_data/train/', sr=12000)
    valDataset = AudioDataset('../../../../audio_data/test/', sr=12000)

    trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = AudioAutoEncoder(input_size=12000)
    ModelSummary(model)

    logger = pl.loggers.TensorBoardLogger('tb_logs', name='audio_autoencoder')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="model_weights",
        filename="audioautoencoder-model-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = pl.Trainer(accelerator='gpu', devices=1,
                        max_epochs=EPOCHS, logger=logger,
                        callbacks=[checkpoint_callback])

    trainer.fit(model, trainDataLoader, valDataLoader)

    trainer.save_checkpoint('model_weights/final-autoencoder-model.ckpt')
