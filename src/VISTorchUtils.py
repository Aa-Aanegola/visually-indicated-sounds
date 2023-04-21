import os
import pickle
import numpy as np
import glob
import torch
import torch.nn as nn

from typing import Tuple, List
from torch.utils.data import Dataset

from VISDataPoint import VISDataPoint
    
class VISDataset(Dataset):

    materials = ['None',
                 'rock',
                 'leaf',
                 'water',
                 'wood',
                 'plastic-bag',
                 'ceramic',
                 'metal',
                 'dirt',
                 'cloth',
                 'plastic',
                 'tile',
                 'gravel',
                 'paper',
                 'drywall',
                 'glass',
                 'grass',
                 'carpet']

    def __init__(self, root:str) -> None:
        self.root = root
        self.files = glob.glob(os.path.join(self.root, '*.pkl'))

    def __getitem__(self, index) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, int]:
        fileName = os.path.join(self.root, f'{index}.pkl')

        with open(fileName, 'rb') as f: 
            dataPoint: VISDataPoint = pickle.load(f)

        coch = dataPoint.cochleagram
        frames, frame0 = dataPoint.frames

        frames = [frame.transpose(2,0,1).astype(np.float32)/255 for frame in frames]
        frame0 = frame0.transpose(2,0,1).astype(np.float32)/255

        material = dataPoint.material

        return coch, frames, frame0, VISDataset.materials.index(material)

    def __len__(self) -> int:
        return len(self.files)


class VISLoss(nn.Module):
    
    def __init__(self, epsilon:float=25**-2) -> None:
        super().__init__()

        self.epsilon = epsilon
    
    def rho(self, r:torch.Tensor) -> torch.Tensor:
        return torch.log(self.epsilon + r**2)

    def forward(self, output, target):
        # output: batchx42x45
        # target: batchx42x45  
        loss = torch.zeros(output.shape[0], device=output.device)

        # E = sum over T (rho(||st - st_hat||))
        for i in range(output.shape[2]):
            r = output[:,:,i] - target[:,:,i]
            r = torch.linalg.vector_norm(r, dim=1)
            loss += self.rho(r)

        return loss.mean()
    
    
class AudioDataset(Dataset):
    def __init__(self, root : str, freq=12000):
        self.root = root
        self.files = glob.glob(os.path.join(self.root, '*.pkl'))
        self.freq = freq
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fileName = os.path.join(self.root, f'{idx}.pkl')
        
        with open(fileName, 'rb') as f:
            wav = pickle.load(f)
        
        downsampled = wav[::wav.shape[0]//self.freq]
        
        return torch.tensor(downsampled)