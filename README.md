# How to start

<details>
  <summary><h2>Instructions how to install PhaseHunter</h2></summary>
1. Install Anaconda or Miniconda: If you haven't installed Anaconda or Miniconda, download and install it from [Anaconda's official website](https://www.anaconda.com/download) or [Miniconda's official website](https://docs.conda.io/projects/miniconda/en/latest/) respectively.

2. Create a New Environment: Open a terminal or Anaconda prompt and run the following command to create a new environment named phasehunter (you can choose a different name if you want):
```bash
conda create --name phasehunter python=3.10
```

3. Activate the New Environment:
```bash
conda activate phasehunter
```

4. Install PyTorch and torchvision: Based on your system and CUDA version, install PyTorch and torchvision. Instructions can be found on the [PyTorch official site](https://pytorch.org/get-started/locally/). For example, for Linux use:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

5. Install Required Libraries:
```bash
conda install numpy pandas scikit-learn scipy tqdm
pip install obspy pytorch-lightning lightning wandb
pip install git+https://github.com/nikitadurasov/masksembles
```

6. Install PhaseHunter
```bash
pip install git+https://github.com/crimeacs/PhaseHunter
```

7. Final Notes:

* Remember to activate your environment (`conda activate phasehunter`) every time you work on this project.
* It's a good idea to periodically update the packages to get the latest bug fixes and improvements.
* If you encounter any compatibility issues or errors during installation, they might be due to version conflicts. In that case, you'll need to identify the versions that are compatible and specify them during installation.

You should now have a working conda environment with the necessary packages for your project!
</details>

## How to use PhaseHunter
PhaseHunter is designed to be straightforward to use, even for those new to seismic data processing. Here's a step-by-step guide:

1. Load the Pre-trained Model:
First, import PhaseHunter
```python
import torch
from phasehunter.model import PhaseHunter
```

Then load the pre-trained PhaseHunter model from a checkpoint (if on Linux):

```python
# Auto select device, note that PhaseHunter works best on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PhaseHunter.load_from_checkpoint('ckpts/30s_STEAD_decay-epoch=196.ckpt')
model = model.eval().to(device)
```

<details>
  <summary>If on Mac, please use:</summary>

  ```python
  device = torch.device("mps")
  
  model = PhaseHunter.load_from_checkpoint('ckpts/30s_STEAD_decay-epoch=196.ckpt')
  model = model.eval().float().to(device)
  ```
</details>


At this time, 2 pre-trained models are available:

* `ckpts/30s_STEAD_decay-epoch=196.ckpt` - 30s version of PhaseHunter trained on STEAD dataset for 200 epochs
* `ckpts/30s_ETHZ_decay-epoch=187.ckpt` - 30s version of PhaseHunter trained on ETHZ dataset for 200 epochs

2. Download or use your own waveform data, e.g.:

```python
from obspy import read
st = read('path_to_your_waveform_data')
predictions = model.process_continuous_waveform(st)
print(predictions)
```

`process_continuous_waveform` works with 3 channel Obspy streams of any length

## For a more comprehansive tutorial try PhaseHunter in Google Colab
<a target="_blank" href="https://colab.research.google.com/github/crimeacs/PhaseHunter/blob/main/PhaseHunter_intro.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
