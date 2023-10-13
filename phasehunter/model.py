from typing import Optional, Union, Tuple, Any
import math

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

from obspy import Stream

seed_everything(42, workers=False)
torch.set_float32_matmul_precision('medium')

class BlurPool1D(nn.Module):
    """Implements 1D version of blur pooling.
    
    Attributes:
        channels (int): Number of input channels.
        pad_type (str): Type of padding (reflect, replicate, zero).
        filt_size (int): Filter size for blur pooling.
        stride (int): Stride size for downsampling.
        pad_off (int): Padding offset.
    """
    def __init__(self, channels: int, pad_type: str='reflect', filt_size: int=3, stride: int=2, pad_off: int=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        # Calculate padding sizes for the beginning and end of signal
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels
        
        # Generate coefficients for the specified filter size using binomial coefficients
        a = np.array([comb(filt_size-1, i, exact=False) for i in range(filt_size)])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)  # normalize the filter
        # Make the filter to have same size with number of channels
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        # Get the appropriate padding layer
        self.pad = self.get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        """Computes forward pass for blur pooling."""
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride]
            else:
                # Apply padding if pad_off is not zero
                return self.pad(inp)[:, :, ::self.stride]
        else:
            # Convolve input with filter and then apply downsampling
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

    def get_pad_layer_1d(self, pad_type: str):
        """Returns appropriate padding layer based on the pad_type string.
        
        Args:
            pad_type: Type of padding. It can be 'refl', 'reflect', 'repl', 'replicate', or 'zero'.
            
        Returns:
            Appropriate padding layer based on pad_type.
        
        Raises:
            ValueError: If pad_type is not recognized.
        """
        # Define the padding layer depending on the input pad_type
        if pad_type in ['refl', 'reflect']:
            pad_layer = nn.ReflectionPad1d
        elif pad_type in ['repl', 'replicate']:
            pad_layer = nn.ReplicationPad1d
        elif pad_type == 'zero':
            pad_layer = nn.ZeroPad1d
        else:
            # Raise an error if pad_type is not recognized
            raise ValueError(f"Pad type [{pad_type}] not recognized")
        return pad_layer


class Masksembles1D(nn.Module):
    """Implements 1D version of Masksembles operation.
    
    Masksembles operation applies different masks to the input in a way that allows the model to estimate uncertainty and confidence at inference time.
    
    Attributes:
        channels (int): Number of input channels.
        n (int): Number of masks to generate.
        scale (float): Scaling factor for masks.
    """
    def __init__(self, channels: int, n: int, scale: float):
        super().__init__()

        self.channels = channels
        self.n = n
        self.scale = scale

        # Generate masks using a provided function
        masks = common.generation_wrapper(channels, n, scale)
        masks = torch.from_numpy(masks)
        
        # Convert masks into PyTorch Parameter and set it to not require gradient
        self.masks = torch.nn.Parameter(masks, requires_grad=False)

    def forward(self, inputs):
        """Computes forward pass for Masksembles operation.
        
        The input is divided into multiple groups, each group is multiplied with a different mask, and then the results
        are concatenated together.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying Masksembles operation.
        """
        # Number of samples in the batch
        batch = inputs.shape[0]
        
        # Divide the input into n groups along the batch dimension
        x = torch.split(inputs.unsqueeze(1), batch // self.n, dim=0)
        
        # Concatenate the groups along the new dimension and permute the dimensions
        x = torch.cat(x, dim=1).permute([1, 0, 2, 3])
        
        # Multiply each group with a different mask
        x = x * self.masks.unsqueeze(1).unsqueeze(-1)
        
        # Concatenate the results along the channel dimension
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        
        # Remove the extra dimension and convert the tensor to the original data type
        return x.squeeze(0).type(inputs.dtype)


class BasicBlock(nn.Module):
    """Implements a basic block of convolutions, a fundamental part of PhaseHunter.
    
    A basic block consists of two convolutional layers, each followed by batch normalization. The output from the second 
    convolutional layer is added to the shortcut connection before applying an optional activation function.
    
    Attributes:
        in_planes (int): Number of input channels (also known as input planes).
        planes (int): Number of output channels (also known as output planes or filters).
        stride (int, optional): Stride size for convolution. Default is 1.
        kernel_size (int, optional): Kernel size for convolution. Default is 7.
        groups (int, optional): Number of groups for convolution. Default is 1.
        do_activation (bool, optional): Whether to apply an activation function (ReLU) at the end. Introduced for embedding capture. Default is True.
    """
    def __init__(self, in_planes: int, planes: int, stride: int = 1, kernel_size: int = 7, groups: int = 1, do_activation: bool = True):
        super(BasicBlock, self).__init__()
        
        self.do_activation = do_activation

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        # Shortcut connection, used to match the dimensionality between input and output
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, padding='same', bias=False),
            nn.BatchNorm1d(planes)
        )

    def forward(self, x):
        """Computes forward pass for the block.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the basic block.
        """
        # Apply first convolution followed by ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Apply second convolution
        out = self.bn2(self.conv2(out))
        
        # Add the output of the shortcut connection
        out += self.shortcut(x)
        
        # Apply activation (it's here for the embedding)
        if self.do_activation:
            out = F.relu(out)
            
        return out



class PhaseHunter(pl.LightningModule):
    """Implements PhaseHunter model for seismic phase picking.
    
    Attributes:
        n_masks (int): Number of masks for Masksembles operation.
        n_outs (int): Number of output units.
    """
    def __init__(self, n_masks=128, n_outs=2):
        super().__init__()

        self.n_masks = 128
        self.n_outs = n_outs

        # Define sequential layers for block 1 to 9
        # Each block consist of BasicBlock, GELU activation, BlurPool1D, and GroupNorm layers
        # Blocks vary in the number of in and out features
        
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
            BasicBlock(1024,128, kernel_size=7, groups=128, do_activation=False),

            # Works better with those off on the last layer before regressor
            # BlurPool1D(512, filt_size=3, stride=2),
            # nn.GELU(),
            # nn.GroupNorm(2,512),
        )

        # Final output layer with Sigmoid activation
        self.out = nn.Sequential(
            nn.LazyLinear(n_outs),
            nn.Sigmoid()
        )

        # Save hyperparameters and initialize Mean Absolute Error loss
        self.save_hyperparameters(ignore=['picker'])
        self.mae = MeanAbsoluteError()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward pass for the model."""
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
        x = self.out(F.relu(embedding))
        
        return x, embedding
        
    def compute_loss(self, y: torch.Tensor, pick: torch.Tensor, mae_name: Optional[Union[str, bool]] = False) -> torch.Tensor:
        """Computes loss for the predictions.
        
        Args:
            y (torch.Tensor): The ground truth tensor.
            pick (torch.Tensor): The predicted tensor.
            mae_name (Union[str, bool], optional): The name for the Mean Absolute Error (MAE) metric. 
                If provided, it logs the MAE metric with the name 'MAE/{mae_name}_val'. Default is False.
    
        Returns:
            torch.Tensor: The computed loss.
        """
        # Filter non-zero values
        y_filt = y[y != 0]
        pick_filt = pick[y != 0]
    
        # Compute L1 loss if there are non-zero values
        if len(y_filt) > 0:
            loss = F.l1_loss(y_filt, pick_filt.flatten())
    
            # If mae_name is provided, log the MAE metric
            if mae_name != False:
                mae_phase = self.mae(y_filt, pick_filt.flatten())*30
                self.log(f'MAE/{mae_name}_val', mae_phase,  on_step=False, on_epoch=True, prog_bar=False)
        else:
            loss = 0
        return loss
    
    def get_likely_val(self, array: np.ndarray) -> Tuple[np.ndarray, gaussian_kde, torch.Tensor, float]:
        """Computes most likely value using Kernel Density Estimation.
        
        Args:
            array (np.ndarray): The input array for which to compute the most likely value.
    
        Returns:
            Tuple[np.ndarray, gaussian_kde, torch.Tensor, float]: A tuple containing 
                - the distribution space (dist_space), 
                - the Kernel Density Estimation (kde), 
                - the most likely value (val), and 
                - the uncertainty of the estimation.
        """
        # Compute KDE for the input array
        kde = gaussian_kde(array)
        
        # Define the distribution space
        dist_space = np.linspace(min(array)-0.001, max(array)+0.001, 512)
    
        # Compute the most likely value and the uncertainty
        val = torch.tensor(dist_space[np.argmax(kde(dist_space))], dtype=torch.float32)
        uncertainty = dist_space.ptp()/2
    
        return dist_space, kde, val, uncertainty

    def process_continuous_waveform(self, st: Stream) -> pd.DataFrame:
        """
        Processes a continuous seismic waveform and predicts P and S wave arrival times using PhaseHunter.

        Parameters:
        -----------
        st : Stream
            The input seismic data as an ObsPy Stream object with three components.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the following columns:
                - p_time: Predicted P-wave arrival time.
                - s_time: Predicted S-wave arrival time.
                - p_uncert: Uncertainty associated with the P-wave prediction.
                - s_uncert: Uncertainty associated with the S-wave prediction.
                - embedding: Embedding representation of the chunk.
                - p_conf: Confidence level of the P-wave prediction.
                - s_conf: Confidence level of the S-wave prediction.
                - p_time_rel: Relative P-wave arrival time in seconds from the start of the input stream.
                - s_time_rel: Relative S-wave arrival time in seconds from the start of the input stream.

        Notes:
        ------
        The function assumes that the input Stream object has three components.
        The neural network inference is performed on chunks of data of 30 seconds. 
        The output DataFrame is a result of aggregating predictions for each chunk and filtering duplicate rows.

        Raises:
        -------
        AssertionError
            If the input Stream object doesn't contain three components.

        Examples:
        ---------
        >>> from obspy import read
        >>> st = read('path_to_your_waveform_data')
        >>> predictions = process_continuous_waveform(st)
        >>> print(predictions)
        """
        assert len(st) == 3, 'For the moment, PhaseHunter works only with 3C input data'
        
        start_time = st[0].stats.starttime
        end_time = st[0].stats.endtime
        
        chunk_size = 30
        
        chunks = []
        predictions = pd.DataFrame()
        
        for chunk_start in tqdm(np.arange(start_time, end_time, chunk_size)):
            chunk_end = chunk_start + chunk_size
    
            chunk = st.slice(chunk_start, chunk_end)
            chunk_orig = np.vstack([x.data for x in chunk], dtype='float')[:,:-1]
            
            if chunk_orig.shape[-1] != chunk_size * 100:
                continue
            
            chunk = chunk_orig - chunk_orig.mean(axis=0)
            max_val = np.max(np.abs(chunk))
            chunk = chunk/max_val
    
            chunk = torch.tensor(chunk, dtype=torch.float)
    
            inference_sample = torch.stack([chunk]*128).to(self.device)
            
            with torch.no_grad():
                preds, embeddings = self(inference_sample)
    
                p_pred = preds[:,0].detach().cpu()
                s_pred = preds[:,1].detach().cpu()
                embeddings = torch.mean(embeddings, axis=0).detach().cpu().numpy()

                p_dist, p_kde, p_val, p_uncert = self.get_likely_val(p_pred)
                s_dist, s_kde, s_val, s_uncert = self.get_likely_val(s_pred)
    
                p_time = chunk_start+p_val.item()*chunk_size
                s_time = chunk_start+s_val.item()*chunk_size
                
                current_predictions = pd.DataFrame({'p_time': p_time, 's_time':s_time,
                                                    'p_uncert' : p_uncert, 's_uncert' : s_uncert,
                                                    'embedding' : [embeddings]})

                predictions = pd.concat([predictions, current_predictions], ignore_index=True)
        
        predictions = predictions.drop_duplicates(subset=['p_uncert', 's_uncert']).reset_index()  

        predictions['p_conf'] = 1/predictions['p_uncert']
        predictions['s_conf'] = 1/predictions['s_uncert']
    
        predictions['p_conf'] /= predictions['p_conf'].max()
        predictions['s_conf'] /= predictions['s_conf'].max()
    
        predictions['p_time_rel'] = (predictions.p_time.apply(lambda x: pd.Timestamp(x.timestamp, unit='s')) - pd.Timestamp(predictions.p_time.iloc[0].date)).dt.total_seconds()
        predictions['s_time_rel'] = (predictions.s_time.apply(lambda x: pd.Timestamp(x.timestamp, unit='s')) - pd.Timestamp(predictions.s_time.iloc[0].date)).dt.total_seconds()
    
        return predictions
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Defines a single step in the training loop for PhaseHunter.
    
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing an input batch (x), 
                and the corresponding P-wave (y_p) and S-wave (y_s) target tensors.
            batch_idx (int): The index of the current batch.
    
        Returns:
            torch.Tensor: The computed loss for this training step.
        """
        # Unpack the batch
        x, y_p, y_s = batch
    
        # Perform forward pass and get predictions
        picks, embedding = self(x)
    
        # Extract P and S phase picks
        p_pick  = picks[:,0]
        s_pick  = picks[:,1]
    
        # Compute losses for P and S phase picks
        p_loss = self.compute_loss(y_p, p_pick, mae_name='P')
        s_loss = self.compute_loss(y_s, s_pick, mae_name='S')
    
        # Combine losses
        loss = (p_loss+s_loss)/self.n_outs
    
        # Log the loss
        self.log('Loss/train', loss, on_step=True, on_epoch=False, prog_bar=True)
    
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Defines a single step in the validation loop for PhaseHunter.
    
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing an input batch (x), 
                and the corresponding P-wave (y_p) and S-wave (y_s) target tensors.
            batch_idx (int): The index of the current batch.
    
        Returns:
            torch.Tensor: The computed loss for this validation step.
        """
        # Unpack the batch
        x, y_p, y_s = batch
    
        # Perform forward pass and get predictions
        picks, embedding = self(x)
    
        # Extract P and S phase picks
        p_pick  = picks[:,0]
        s_pick  = picks[:,1]
    
        # Compute losses for P and S phase picks
        p_loss = self.compute_loss(y_p, p_pick, mae_name='P')
        s_loss = self.compute_loss(y_s, s_pick, mae_name='S')
    
        # Combine losses
        loss = (p_loss+s_loss)/self.n_outs
    
        # Log the loss
        self.log('Loss/val',  loss, on_step=False, on_epoch=True, prog_bar=False)
    
        return loss
    
    # def configure_optimizers(self) -> dict:
    #     """
    #     Defines the optimizer and scheduler for PhaseHunter.
    
    #     Returns:
    #         dict: A dictionary containing the optimizer, the learning rate scheduler, and the metric to monitor.
    #     """
    #     # Define the optimizer
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    #     # Define the learning rate scheduler
    #     # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, cooldown=10, threshold=1e-6)
    
    #     # Define the metric to monitor
    #     # monitor = 'Loss/train'
    
    #     return {"optimizer": optimizer}#,  "lr_scheduler": scheduler, 'monitor': monitor}

    def configure_optimizers(self) -> dict:
        """
        Defines the optimizer and scheduler for PhaseHunter.
    
        Returns:
            dict: A dictionary containing the optimizer, the learning rate scheduler, and the metric to monitor.
        """
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
        # Total number of epochs for decay
        decay_epochs = 100
    
        # Total number of epochs including constant learning rate period
        total_epochs = 200
    
        # Final learning rate
        final_lr = 1e-7
    
        # Lambda function for learning rate schedule
        def lambda_func(epoch):
            if epoch < decay_epochs:
                return 1.0  # constant learning rate
            else:
                epoch_adjusted = epoch - decay_epochs
                return 1 - epoch_adjusted/decay_epochs + (final_lr/1e-3)*epoch_adjusted/decay_epochs
    
        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
    
        # Define the metric to monitor
        # monitor = 'Loss/train'
    
        return {"optimizer": optimizer,  "lr_scheduler": scheduler}

    
    