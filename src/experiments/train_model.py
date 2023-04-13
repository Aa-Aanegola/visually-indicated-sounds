import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from VISTorchUtils import VISDataset, VISLoss
from VISDataPoint import VISDataPoint
from utils import visCollate


trainDataset = VISDataset('/scratch/vis_data/train')
valDataset = VISDataset('/scratch/vis_data/test')


BATCH_SIZE=16


trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, collate_fn=visCollate, shuffle=True, num_workers=6)
valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, collate_fn=visCollate, num_workers=6)


class VISTrivialModel(pl.LightningModule):

    def __init__(self, outputSize:int):
        super().__init__()
        self.featureExtractor = resnet18(ResNet18_Weights.DEFAULT)
        self.featureExtractor.fc = nn.Identity()

        for param in self.featureExtractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, outputSize)


    def forward(self, stFrames, frame0):

        # stFrames: batchx45x224x224x3
        # frame0: batchx224x224x3

        stFrameFeatures = []
        for i in range(stFrames.shape[1]):
            currStFrame = stFrames[:,i,:,:,:].squeeze(1)
            currStFrameFeatures = self.featureExtractor(currStFrame)
            stFrameFeatures.append(currStFrameFeatures)
        stFrameFeatures = torch.stack(stFrameFeatures, dim=1)

        frame0Features = self.featureExtractor(frame0).unsqueeze(1).repeat(1, stFrames.shape[1], 1)
        # X = torch.cat([stFrameFeatures, frame0Features], dim=2)
        X = stFrameFeatures
        
        X.detach_()
        # X is the input to the LSTM -> batchx45x1024
        X, _ = self.lstm(X)

        out = self.fc(X).transpose(1, 2)
        # # Pass the LSTM output of each timestep through a linear layer
        # out = []

        # for i in range(X.shape[1]):
        #     currOut = self.fc(X[:,i,:])
        #     out.append(currOut)
        
        # out = torch.stack(out, dim=1).transpose(1,2)
        
        return out
    
    def training_step(self, batch, batch_idx):
        coch, stFrames, frame0, material = batch
        out = self(stFrames, frame0)
        # loss = VISLoss()(out, coch)
        loss = nn.MSELoss()(out, coch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        coch, stFrames, frame0, material = batch
        out = self(stFrames, frame0)
        # loss = VISLoss()(out, coch)
        loss = nn.MSELoss()(out, coch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5*1e-5)
        return optimizer


model = VISTrivialModel(outputSize=42)


logger = pl.loggers.TensorBoardLogger('tb_logs', name='trivial_model')


trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=25, logger=logger)


trainer.fit(model, trainDataLoader, valDataLoader)

