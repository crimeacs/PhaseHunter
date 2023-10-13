from lightning.pytorch.loggers import WandbLogger
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

batch_size = 1024
num_workers = 32

model = PhaseHunter(n_masks=128, n_outs=2)#.load_from_checkpoint('ckpts/30sSTEAD-epoch=92.ckpt')

df = pd.read_csv('Seisbench_DATA/stead_mem.csv')#[['trace_P_final','trace_S_final']].fillna(0)

# Get the unique source IDs
unique_source_ids = df['source_id'].unique()

# Split unique source IDs into train, val, and test sets
train_size = int(0.8 * len(unique_source_ids))
val_size = int(0.1 * len(unique_source_ids))

test_ids = unique_source_ids[train_size + val_size:]
unique_source_ids_train_val = unique_source_ids[:train_size + val_size]

# Shuffle the unique source IDs
np.random.shuffle(unique_source_ids_train_val)

train_ids = unique_source_ids_train_val[:train_size]
val_ids = unique_source_ids_train_val[train_size:train_size + val_size]

# Create masks for train, val, and test based on source ID
train_mask = df['source_id'].isin(train_ids)
val_mask = df['source_id'].isin(val_ids)
test_mask = df['source_id'].isin(test_ids)

# Create train, val, and test dataframes
df_train = df[train_mask][['trace_P_final','trace_S_final']].fillna(0)
df_val = df[val_mask][['trace_P_final','trace_S_final']].fillna(0)
df_test = df[test_mask][['trace_P_final','trace_S_final']].fillna(0)

data = np.memmap('Seisbench_DATA/stead.dat', dtype='float32', mode='r', shape=(len(df), 3, 9000))

augmentations = Augmentations(padding=120, crop_length=3000, fs=100, lowcut=0.2, highcut=40, order=4)

train_dataset = Waveforms_dataset(meta=df_train, data=data, test=False, transform=True, augmentations=augmentations)
val_dataset = Waveforms_dataset(meta=df_val, data=data, test=False, transform=True, augmentations=augmentations)
test_dataset = Waveforms_dataset(meta=df_test, data=data, test=False, transform=True, augmentations=augmentations)

train_loader = DataLoader(train_dataset, shuffle=True,  num_workers=num_workers, batch_size=batch_size, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, shuffle=False,  num_workers=num_workers, batch_size=batch_size, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, shuffle=False,  num_workers=num_workers, batch_size=batch_size, pin_memory=True, drop_last=True)

# swa_callback = StochasticWeightAveraging(swa_lrs=0.05)

# Initialize a new wandb run
wandb_logger = WandbLogger(project='PhaseHunter')

# Initialize profiler
profiler = PyTorchProfiler(filename="perf-logs")
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(dirpath='ckpts', filename="30s_STEAD_decay-{epoch:02d}", save_top_k=1, monitor="Loss/val")

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
    
            # limit_train_batches=0.001,
            # limit_val_batches=0.01,
    
            max_epochs=-1,
            # fast_dev_run=True,
        )

if __name__ == "__main__":
    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader,
                ckpt_path='last'
                )
