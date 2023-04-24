import sys

sys.path.append('../..')

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle
import random
from tqdm import tqdm
from scipy.signal import resample

from IPython.display import Audio

torch.set_float32_matmul_precision('medium')



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

class AudioDataset(Dataset):
    def __init__(self, root: str, sr: int=48000):
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

    sampling_rate = 48000
    BATCH_SIZE = 16
    NUM_WORKERS = 15
    EPOCHS = 100

    trainDataset = AudioDataset('../../../../audio_data/train/', sr=sampling_rate)
    valDataset = AudioDataset('../../../../audio_data/test/', sr=sampling_rate)

    trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = AudioAutoEncoderConv(input_size=sampling_rate)
    ModelSummary(model)

    logger = pl.loggers.TensorBoardLogger('tb_logs', name='audio_autoencoder')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="model_weights",
        filename="audioautoencoderconv-model-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = pl.Trainer(accelerator='gpu', devices=1,
                        max_epochs=EPOCHS, logger=logger,
                        callbacks=[checkpoint_callback])

    trainer.fit(model, trainDataLoader, valDataLoader)
    trainer.save_checkpoint('model_weights/final-autoencoderconv-model.ckpt')
