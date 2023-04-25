import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import os

from transformers import VideoMAEForVideoClassification
from typing import Dict, Any


class VISVMAEModel(pl.LightningModule):

    def __init__(self):

        super().__init__()
        os.environ['TORCH_HOME'] = 'scratch/arihanth.srikar/'
        self.featureExtractor = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
        self.featureExtractor.classifier = nn.Identity()

        for param in self.featureExtractor.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1890),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.featureExtractor(X).logits
        X.detach_()

        out = self.mlp(X).reshape(-1, 42, 45)
        return out
    
    def _common_step(self, batch, batch_idx):
        coch, frames, material = batch
        out = self(frames)
        loss = nn.MSELoss()(out, coch)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)

        modified_state_dict = {}
        for k, v in self.mlp.state_dict().items():
            modified_state_dict[f'mlp.{k}'] = v

        checkpoint['state_dict'] = modified_state_dict

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        
        modified_state_dict = {}
        for k, v in self.featureExtractor.state_dict().items():
            modified_state_dict[f'featureExtractor.{k}'] = v
            
        checkpoint['state_dict'].update(modified_state_dict)
        return super().on_load_checkpoint(checkpoint)        

class VISVMAEModelFT(pl.LightningModule):

    def __init__(self, unfreezeEpoch):

        super().__init__()
        self.featureExtractor = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
        self.featureExtractor.classifier = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1890),
            nn.Sigmoid()
        )
        self.unfreezeEpoch = unfreezeEpoch
        self.freezeFeatureExtractor()

    def freezeFeatureExtractor(self):
        for param in self.featureExtractor.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreezeFeatureExtractor(self):
        for param in self.featureExtractor.parameters():
            param.requires_grad = True
        self.frozen = False

    def forward(self, X):
        X = self.featureExtractor(X).logits

        if self.frozen:
            X.detach_()

        out = self.mlp(X).reshape(-1, 42, 45)
        return out
    
    def _common_step(self, batch, batch_idx):

        if self.frozen and self.current_epoch == self.unfreezeEpoch:
            self.unfreezeFeatureExtractor()
            print("Reached unfreeze epoch, unfreezing feature extractor")

        coch, frames, material = batch
        out = self(frames)
        loss = nn.MSELoss()(out, coch)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)