import sys

sys.path.append('../..')

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint

from VISAudioAutoEncoder import AudioAutoEncoderConv
from VISTorchUtils import AudioDataset

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
