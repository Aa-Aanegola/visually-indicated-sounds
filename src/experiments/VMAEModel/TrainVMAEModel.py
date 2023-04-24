import sys
sys.path.append('../..')
import os

from utils import visCollateV2
from VISDataPoint import VISDataPointV2
from VISTorchUtils import VISDatasetV2
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from VISVMAEModel import VISVMAEModel

os.environ['TORCH_HOME'] = 'scratch/arihanth.srikar/'
BATCH_SIZE = 8
NUM_WORKERS = 15
EPOCHS = 100

trainDataset = VISDatasetV2('/scratch/vis_data_v2/train')
valDataset = VISDatasetV2('/scratch/vis_data_v2/test')

trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, collate_fn=visCollateV2, shuffle=True, num_workers=NUM_WORKERS)
valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, collate_fn=visCollateV2, num_workers=NUM_WORKERS)

model = VISVMAEModel()

logger = pl.loggers.TensorBoardLogger('tb_logs', name='VMAEModel')

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=EPOCHS, logger=logger)

trainer.fit(model, trainDataLoader, valDataLoader)
