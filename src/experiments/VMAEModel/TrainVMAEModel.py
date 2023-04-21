import sys
sys.path.append('../..')

from utils import visCollateV2
from VISDataPoint import VISDataPointV2
from VISTorchUtils import VISDatasetV2
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from VISVMAEModel import VISVMAEModel

trainDataset = VISDatasetV2('/scratch/vis_data_v2/train')
valDataset = VISDatasetV2('/scratch/vis_data_v2/test')

BATCH_SIZE = 8

trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, collate_fn=visCollateV2, shuffle=True, num_workers=6)
valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, collate_fn=visCollateV2, num_workers=6)

model = VISVMAEModel()

logger = pl.loggers.TensorBoardLogger('tb_logs', name='VMAEModel')

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=20, logger=logger)

trainer.fit(model, trainDataLoader, valDataLoader)
