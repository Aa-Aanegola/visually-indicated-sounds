import sys
sys.path.append('../..')


from utils import visCollate
from VISDataPoint import VISDataPoint
from VISTorchUtils import VISDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from VISPaperModel import VISPaperModel

trainDataset = VISDataset('/scratch/vis_data/train')
valDataset = VISDataset('/scratch/vis_data/test')

BATCH_SIZE = 16
NUM_WORKERS = 8
EPOCHS = 100

trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, collate_fn=visCollate, shuffle=True, num_workers=NUM_WORKERS)
valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, collate_fn=visCollate, num_workers=NUM_WORKERS)

model = VISPaperModel(outputSize=42)

logger = pl.loggers.TensorBoardLogger('tb_logs', name='PaperModel')
checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="model_weights",
        filename="paper-mse-{epoch:02d}-{val_loss:.2f}",
    )
trainer = pl.Trainer(accelerator='gpu', devices=1,
                     max_epochs=EPOCHS, logger=logger,
                     callbacks=[checkpoint_callback])

trainer.fit(model, trainDataLoader, valDataLoader)
