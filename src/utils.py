import os
import sys
import numpy as np
import torch

from pycochleagram.cochleagram import invert_cochleagram
from torchvision import transforms
from tqdm import tqdm

from joblib import Parallel, delayed
from typing import List

class NoStdStreams(object):
    """
    Utility class to silence all output from the code within a block. Usage:

    with NoStdStreams():
        *code here*
    """
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

def waveFromCochleagram(cochleagram:np.ndarray):

    with NoStdStreams():
        wave = invert_cochleagram(cochleagram, 96000, 40, 100, 10000, 1, downsample=90, nonlinearity='power')[0]
    return wave

def batchWaveFromCochleagram(cochleagrams:List[np.ndarray]):
    return Parallel(n_jobs=-1)(delayed(waveFromCochleagram)(coch) for coch in tqdm(cochleagrams))

def visCollate(batch):
    cochBatch = []
    stFramesBatch = []
    frame0Batch = []
    materialBatch = []

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)

    for sample in batch:
        coch, stFrames, frame0, material = sample
        cochBatch.append(torch.from_numpy(coch).float())
        stFramesTensor = torch.stack([normalize(torch.from_numpy(frame)) for frame in stFrames])
        stFramesBatch.append(stFramesTensor)
        frame0Batch.append(normalize(torch.from_numpy(frame0)))
        materialBatch.append(material)

    return torch.stack(cochBatch), torch.stack(stFramesBatch), torch.stack(frame0Batch), torch.tensor(materialBatch)

def visCollateV2(batch):
    cochBatch = []
    stackBatch = []
    materialBatch = []

    for sample in batch:
        coch, stack, material = sample
        cochBatch.append(torch.from_numpy(coch).float())
        stackBatch.append(torch.from_numpy(stack))
        materialBatch.append(material)
    cochBatch = torch.stack(cochBatch)
    stackBatch = torch.stack(stackBatch)
    materialBatch = torch.tensor(materialBatch)

    return cochBatch, stackBatch, materialBatch