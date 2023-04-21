import sys
sys.path.append('../..')


from utils import visCollate
from VISDataPoint import VISDataPoint
from VISTorchUtils import VISDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from experiments.BiLSTMModel.VISBiLSTMModel import VISBiLSTMModel

trainDataset = VISDataset('/scratch/vis_data/train')
valDataset = VISDataset('/scratch/vis_data/test')

BATCH_SIZE = 16
NUM_WORKERS = 15

trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, collate_fn=visCollate, shuffle=True, num_workers=NUM_WORKERS)
valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, collate_fn=visCollate, num_workers=NUM_WORKERS)

model = VISBiLSTMModel(outputSize=42, isBi=True, hidden_dim=256)

# saves top-K checkpoints based on "val_loss" metric
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    dirpath="model_weights",
    filename="bilstm-model-{epoch:02d}-{val_loss:.2f}",
)

logger = pl.loggers.TensorBoardLogger('tb_logs', name='bilstm_model')

trainer = pl.Trainer(accelerator='gpu', devices=1,
                     max_epochs=100, logger=logger,
                     callbacks=[checkpoint_callback])

trainer.fit(model, trainDataLoader, valDataLoader)

trainer.save_checkpoint('model_weights/final-bilstm-mode.ckpt')
