from lightning import seed_everything
import lightning as pl

from masksembles import common
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scipy.stats import gaussian_kde
from scipy.special import comb

from tqdm.auto import tqdm
import pandas as pd

seed_everything(42, workers=False)
torch.set_float32_matmul_precision('medium')

class BlurPool1D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels
        
        # Generate coefficients for the specified filter size
        a = np.array([comb(filt_size-1, i, exact=False) for i in range(filt_size)])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print(f"Pad type [{pad_type}] not recognized")
    return PadLayer

class Masksembles1D(nn.Module):

    def __init__(self, channels: int, n: int, scale: float):
        super().__init__()

        self.channels = channels
        self.n = n
        self.scale = scale

        masks = common.generation_wrapper(channels, n, scale)
        masks = torch.from_numpy(masks)
        
        self.masks = torch.nn.Parameter(masks, requires_grad=False)

    def forward(self, inputs):
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.n, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2, 3])
        x = x * self.masks.unsqueeze(1).unsqueeze(-1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        
        return x.squeeze(0).type(inputs.dtype)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size=7, groups=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, padding='same', bias=False),
            nn.BatchNorm1d(self.expansion*planes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PhaseHunter(pl.LightningModule):
    def __init__(self, n_masks=128, n_outs=2):
        super().__init__()

        self.n_masks = 128
        self.n_outs = n_outs
        
        self.block1 = nn.Sequential(
            BasicBlock(3,8, kernel_size=7, groups=1),
            nn.GELU(),
            BlurPool1D(8, filt_size=3, stride=2),
            nn.GroupNorm(2,8),
        )
        
        self.block2 = nn.Sequential(
            BasicBlock(8, 16, kernel_size=7, groups=8),
            nn.GELU(),
            BlurPool1D(16, filt_size=3, stride=2),
            nn.GroupNorm(2,16),
        )
        
        self.block3 = nn.Sequential(
            BasicBlock(16,32, kernel_size=7, groups=16),
            nn.GELU(),
            BlurPool1D(32, filt_size=3, stride=2),
            nn.GroupNorm(2,32),
        )
        
        self.block4 = nn.Sequential(
            BasicBlock(32,64, kernel_size=7, groups=32),
            nn.GELU(),
            BlurPool1D(64, filt_size=3, stride=2),
            nn.GroupNorm(2,64),
        )
        
        self.block5 = nn.Sequential(
            BasicBlock(64,128, kernel_size=7, groups=64),
            nn.GELU(),
            BlurPool1D(128, filt_size=3, stride=2),
            nn.GroupNorm(2,128),
        )
        
        self.block6 = nn.Sequential(
            Masksembles1D(128, self.n_masks, 2.0),
            BasicBlock(128,256, kernel_size=7, groups=128),
            nn.GELU(),
            BlurPool1D(256, filt_size=3, stride=2),
            nn.GroupNorm(2,256),
        )
        
        self.block7 = nn.Sequential(
            Masksembles1D(256, self.n_masks, 2.0),
            BasicBlock(256,512, kernel_size=7, groups=256),
            BlurPool1D(512, filt_size=3, stride=2),
            nn.GELU(),
            nn.GroupNorm(2,512),
        )
        
        self.block8 = nn.Sequential(
            Masksembles1D(512, self.n_masks, 2.0),
            BasicBlock(512,1024, kernel_size=7, groups=512),
            BlurPool1D(1024, filt_size=3, stride=2),
            nn.GELU(),
            nn.GroupNorm(2,1024),
        )
        
        self.block9 = nn.Sequential(
            Masksembles1D(1024, self.n_masks, 2.0),
            BasicBlock(1024,128, kernel_size=7, groups=128),
            # BlurPool1D(512, filt_size=3, stride=2),
            # nn.GELU(),
            # nn.GroupNorm(2,512),
        )

        self.out = nn.Sequential(
            nn.LazyLinear(n_outs),
            nn.Sigmoid()
        )

        self.save_hyperparameters(ignore=['picker'])
        self.mae = MeanAbsoluteError()
        
    def forward(self, x):
        # Feature extraction
        x = self.block1(x)    
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)
        
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.block7(x)
        x = self.block8(x)
        
        x = self.block9(x)
        
        # Regressor
        embedding = x.flatten(start_dim=1)
        x = self.out(embedding)
        
        return x, embedding
        
    def compute_loss(self, y, pick, mae_name=False):
        y_filt = y[y != 0]
        pick_filt = pick[y != 0]
        if len(y_filt) > 0:
            loss = F.l1_loss(y_filt, pick_filt.flatten())
            if mae_name != False:
                mae_phase = self.mae(y_filt, pick_filt.flatten())*120
                self.log(f'MAE/{mae_name}_val', mae_phase,  on_step=False, on_epoch=True, prog_bar=False)
        else:
            loss = 0
        return loss

    def get_likely_val(self, array):
        kde = gaussian_kde(array)
        dist_space = np.linspace(min(array)-0.001, max(array)+0.001, 512)
        val = torch.tensor(dist_space[np.argmax(kde(dist_space))], dtype=torch.float32)
        uncertainty = dist_space.ptp()/2
        return dist_space, kde, val, uncertainty

    def process_continuous_waveform(self, st):
        # Get time of the stream
        start_time = st[0].stats.starttime
        end_time = st[0].stats.endtime
        
        # Define the chunk size in seconds
        chunk_size = 60
        
        # Define the array to store chunks
        chunks = []
        
        predictions = pd.DataFrame()
        
        # Loop over the time range with the step equal to chunk_size
        for chunk_start in tqdm(np.arange(start_time, end_time, chunk_size)):
            # Define the chunk end time
            chunk_end = chunk_start + chunk_size
        
            # Slice the stream
            chunk = st.slice(chunk_start, chunk_end)
            chunk = np.vstack([x.data for x in chunk], dtype='float')[:,:-1]
            chunk -= chunk.mean(axis=0)

            # Normalize
            max_val = np.max(np.abs(chunk))
            chunk = chunk/max_val

            chunk = torch.tensor(chunk, dtype=torch.float)
        
            inference_sample = torch.stack([chunk]*128).cuda()
            
            with torch.no_grad():
                preds, embeddings = self(inference_sample)
                p_pred = preds[:,0].detach().cpu()
                s_pred = preds[:,1].detach().cpu()
                embeddings = torch.mean(embeddings, axis=0).detach().cpu().numpy()
        
                p_dist, p_kde, p_val, p_uncert = self.get_likely_val(p_pred)
                s_dist, s_kde, s_val, s_uncert = self.get_likely_val(s_pred)
        
                p_time = chunk_start+p_val.item()*chunk_size
                s_time = chunk_start+s_val.item()*chunk_size
        
                predictions = predictions.append({'p_time': p_time, 's_time':s_time,
                                    'p_uncert' : p_uncert, 's_uncert' : s_uncert,
                                    'embedding' : embeddings}, ignore_index=True)
        
        predictions['p_conf'] = 1/predictions['p_uncert']
        predictions['s_conf'] = 1/predictions['s_uncert']
        predictions['p_conf'] /= predictions['p_conf'].max()
        predictions['s_conf'] /= predictions['s_conf'].max()
        
        predictions['p_time_rel'] = (predictions.p_time.apply(lambda x: pd.Timestamp(x.timestamp, unit='s')) - pd.Timestamp(predictions.p_time.iloc[0].date)).dt.total_seconds()
        
        predictions['s_time_rel'] = (predictions.s_time.apply(lambda x: pd.Timestamp(x.timestamp, unit='s')) - pd.Timestamp(predictions.s_time.iloc[0].date)).dt.total_seconds()
        return predictions

            
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y_p, y_s = batch
        
        picks, embedding = self(x)
        
        p_pick  = picks[:,0]
        s_pick  = picks[:,1]

        p_loss = self.compute_loss(y_p, p_pick, mae_name='P')
        s_loss = self.compute_loss(y_s, s_pick, mae_name='S')

        loss = (p_loss+s_loss)/self.n_outs
        
        self.log('Loss/train', loss, on_step=True, on_epoch=False, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, y_p, y_s = batch
        
        picks, embedding = self(x)
        
        p_pick  = picks[:,0]
        s_pick  = picks[:,1]

        p_loss = self.compute_loss(y_p, p_pick, mae_name='P')
        s_loss = self.compute_loss(y_s, s_pick, mae_name='S')

        loss = (p_loss+s_loss)/self.n_outs
            
        self.log('Loss/val',  loss, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, cooldown=10, threshold=1e-6)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 3e-4,  epochs=300, steps_per_epoch=len(train_loader))
        monitor = 'Loss/train'
        return {"optimizer": optimizer,  "lr_scheduler": scheduler, 'monitor': monitor}
    
    