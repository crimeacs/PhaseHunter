from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import RichProgressBar, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.profilers import PyTorchProfiler
import random
import lightning as pl

from PhaseHunter.model import PhaseHunter
from PhaseHunter.dataloader import Augmentations, Waveforms_dataset
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

batch_size = 512
num_workers = 2

model = PhaseHunter(n_masks=128, n_outs=2)

df = pd.read_csv('Seisbench_DATA/stead_mem.csv')[['trace_P_final','trace_S_final']].fillna(0)
df_train, df_temp = train_test_split(df, test_size=0.1, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

data = np.memmap('Seisbench_DATA/stead.dat', dtype='float32', mode='r', shape=(len(df), 3, 9000))

augmentations = Augmentations(padding=120, crop_length=6000, fs=100, lowcut=0.2, highcut=40, order=5)

train_dataset = Waveforms_dataset(meta=df_train, data=data, test=False, transform=True, augmentations=augmentations)
val_dataset = Waveforms_dataset(meta=df_val, data=data, test=False, transform=True, augmentations=augmentations)
test_dataset = Waveforms_dataset(meta=df_test, data=data, test=False, transform=True, augmentations=augmentations)

train_loader = DataLoader(train_dataset, shuffle=True,  num_workers=num_workers, batch_size=batch_size, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, shuffle=True,  num_workers=num_workers, batch_size=batch_size, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, shuffle=True,  num_workers=num_workers, batch_size=batch_size, pin_memory=True, drop_last=True)

# swa_callback = StochasticWeightAveraging(swa_lrs=0.05)

# Initialize a new wandb run
wandb_logger = WandbLogger(project='PhaseHunter')

# Initialize profiler
profiler = PyTorchProfiler(filename="perf-logs")
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(dirpath='ckpts', save_top_k=1, monitor="val_loss")

# # train model
trainer = pl.Trainer(
            precision='16-mixed',
            
            callbacks=[checkpoint_callback, lr_monitor],
            devices='auto', 
            accelerator="auto",
    
            # auto_select_gpus=True,
            # strategy=DDPStrategy(find_unused_parameters=False),
            benchmark=True,
            
            gradient_clip_val=0.5,
    
            logger=wandb_logger,
            log_every_n_steps=50,
            enable_progress_bar=True,
    
            limit_train_batches=0.1,
            limit_val_batches=0.1,
    
            max_epochs=1,
            # fast_dev_run=True,
        )

if __name__ == "__main__":
    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader,
                )