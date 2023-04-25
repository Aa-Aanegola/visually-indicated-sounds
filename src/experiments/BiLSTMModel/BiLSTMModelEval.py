import sys
sys.path.append('../..')

import torch
import pytorch_lightning as pl
import numpy as np

from torch.utils.data import DataLoader

from VISTorchUtils import VISDataset, VISLoss
from VISDataPoint import VISDataPoint
from utils import visCollate, batchWaveFromCochleagram
from metrics import Evaluator

from VISBiLSTMModel import VISBiLSTMModel
import matplotlib.pyplot as plt
from tqdm import tqdm

from IPython.display import Audio
valDataset = VISDataset('/scratch/vis_data/test')
BATCH_SIZE = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, collate_fn=visCollate, shuffle=True, num_workers=4)
model = VISBiLSTMModel.load_from_checkpoint('model_weights/final-bilstm-model.ckpt', outputSize=42).to(device)

def infer(batch):
    _, frames, _, _ = batch
    return list(model(frames.to(device)).cpu().detach().numpy())

print("Running Inferences...")
preds = []
gt = []
for batch in tqdm(valDataLoader):
    preds += infer(batch)
    gt += list(batch[0].numpy())

data = [(p,t) for (p,t) in zip(preds, gt)]
np.random.shuffle(data)
preds, gt = zip(*data)
print("Computing Predicted Waves...")
preds = np.array(batchWaveFromCochleagram(preds[:1000]))
print("Computing Ground Truth Waves...")
gt = np.array(batchWaveFromCochleagram(gt[:1000]))

evaluator = Evaluator(gt, preds)
metrics = evaluator.get_metrics()

print(metrics)
print("Done!")
