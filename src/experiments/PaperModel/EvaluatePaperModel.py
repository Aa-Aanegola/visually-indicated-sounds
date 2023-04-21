import sys
sys.path.append('../..')

import torch
import pytorch_lightning as pl
import numpy as np

from torch.utils.data import DataLoader

from VISTorchUtils import VISDataset, VISLoss
from VISDataPoint import VISDataPoint
from utils import visCollate, waveFromCochleagram, batchWaveFromCochleagram
from metrics import Evaluator

from VISPaperModel import VISPaperModel
import matplotlib.pyplot as plt
from tqdm import tqdm

from IPython.display import Audio
valDataset = VISDataset('/scratch/vis_data/test')
BATCH_SIZE = 8

valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, collate_fn=visCollate, shuffle=True, num_workers=4)
model = VISPaperModel.load_from_checkpoint('tb_logs/VL_100/checkpoints/epoch=99-step=125800.ckpt', outputSize=42)


eval = Evaluator(model, valDataLoader, num_batches=125)
eval.get_metrics()