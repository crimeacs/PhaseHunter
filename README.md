# How to start

## Instructions to Create a Conda Environment
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

## 