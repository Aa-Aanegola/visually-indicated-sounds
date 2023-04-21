import os
import sys
import numpy as np
import torch

from pycochleagram.cochleagram import invert_cochleagram
from torchvision import transforms
from utils import waveFromCochleagram, batchWaveFromCochleagram
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, dataloader, num_batches=250):
        self.model = model
        self.dataloader = dataloader
        self.num_batches = num_batches
        
    def get_metrics(self):
        wavs = []
        for i, data in tqdm(enumerate(self.dataloader)):
            coch, stFrames, frame0, material = data
            out = self.model(stFrames, frame0).detach().cpu().numpy() 
            
            ret = batchWaveFromCochleagram(out)
            wavs.extend(ret)
            print(len(wavs))
            
            if i >= self.num_batches-1:
                break
        
        print(self.reconstructable(wavs))
            
    def reconstructable(self, wavs):
        not_reconstructable = 0
        for wav in wavs:
            if np.isnan(wav).any():
                not_reconstructable += 1
        print(not_reconstructable, len(wavs))
        return 1 - not_reconstructable/len(wavs)