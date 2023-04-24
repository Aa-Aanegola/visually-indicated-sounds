import sys

sys.path.append('../..')

import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle

from torch.utils.data import DataLoader

from VISTorchUtils import VISDatasetV2
from utils import visCollateV2, batchWaveFromCochleagram
from tqdm import tqdm


from VISVMAEModel import VISVMAEModel
from metrics import Evaluator

BATCH_SIZE=4

try:
    with open('cochs.pkl', 'rb') as f:
        gt, preds = pickle.load(f)
except FileNotFoundError:
    print("Cochleagrams not found. Running inferences...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    valDataset = VISDatasetV2('/scratch/vis_data_v2/test')
    valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, collate_fn=visCollateV2, num_workers=4)

    model = VISVMAEModel.load_from_checkpoint('model_weights/vmae-model-epoch=04-val_loss=0.00.ckpt').to(device)

    def infer(batch):
        _, frames, _ = batch
        return list(model(frames.to(device)).cpu().detach().numpy())  

    print("Running Inferences...")

    preds = []
    gt = []
    for batch in tqdm(valDataLoader):
        preds += infer(batch)
        gt += list(batch[0].numpy())

    with open('cochs.pkl', 'wb') as f:
        pickle.dump((gt, preds), f)

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
