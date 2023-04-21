import sys
sys.path.append('../..')


from utils import visCollate
from VISDataPoint import VISDataPoint
from VISTorchUtils import VISDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from VISPaperModel import VISPaperModel

trainDataset = VISDataset('/scratch/vis_data/train')
valDataset = VISDataset('/scratch/vis_data/test')

BATCH_SIZE = 16

trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, collate_fn=visCollate, shuffle=True, num_workers=4)
valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, collate_fn=visCollate, num_workers=4)

model = VISPaperModel(outputSize=42)

logger = pl.loggers.TensorBoardLogger('tb_logs', name='trivial_model')

trainer = pl.Trainer(accelerator='gpu', devices=1,
                     max_epochs=100, logger=logger)

trainer.fit(model, trainDataLoader, valDataLoader)
