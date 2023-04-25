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
from transformers import VideoMAEForVideoClassification
from typing import Dict, Any

from VISAudioAutoEncoder import VISLatentVMAEModel, AudioAutoEncoderConv
from VISTorchUtils import VISDatasetV3


BATCH_SIZE = 16
NUM_WORKERS = 8

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    
    aae = AudioAutoEncoderConv()
    aae = aae.load_from_checkpoint('./model_weights/audioautoencoderconvsm-model-epoch=90-val_loss=0.00.ckpt')
    model = VISLatentVMAEModel(aae)
    
    trainDataset = VISDatasetV3('/scratch/vis_data_v3/train', sr=48000)
    valDataset = VISDatasetV3('/scratch/vis_data_v3/test', sr=48000)
    
    trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    logger = pl.loggers.TensorBoardLogger('tb_logs', name='audio_autoencoder')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="model_weights",
        filename="vmae-aae-model-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = pl.Trainer(accelerator='gpu', devices=1,
                        max_epochs=15, logger=logger,
                        callbacks=[checkpoint_callback])
    
    trainer.fit(model, trainDataLoader, valDataLoader)
