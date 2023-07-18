# PhaseHunter: Seismic Phase Arrival Time Prediction
PhaseHunter is a Python module for estimating the most likely arrival times of seismic phases in continuous waveform data. This tutorial will guide you through the process of processing waveform data and making phase arrival time predictions using PhaseHunter.

## Installation

Before we start, make sure to install all the necessary dependencies. PhaseHunter primarily depends on `NumPy`, `pandas`, `torch`, `scipy`, and `obspy`. 

To install PhaseHunter:

```shell
pip install git+https://github.com/crimeacs/PhaseHunter.git
```

## Processing continuous waveform data

The main method provided by PhaseHunter is `process_continuous_waveform`. This method takes as input a continuous waveform and makes phase arrival time predictions. The method is part of the PhaseHunter class, and it is defined as follows:

```python
def process_continuous_waveform(self, st: Stream) -> pd.DataFrame:
```

The input, `st`, should be a three-component continuous waveform. 

The method returns a pandas DataFrame with the phase predictions and their uncertainties.

### Method usage

Let's assume that `st` is your seismic stream. Then, the usage is as follows:

```python
# Instantiate PhaseHunter
from PhaseHunter.model import PhaseHunter
ph = PhaseHunter()

# Process waveform
predictions = ph.process_continuous_waveform(st)
```

The returned DataFrame, `predictions`, includes the following columns:

- `p_time`: The most likely P-wave arrival time
- `s_time`: The most likely S-wave arrival time
- `p_uncert`: The uncertainty of the P-wave arrival time
- `s_uncert`: The uncertainty of the S-wave arrival time
- `embedding`: The embedding representation of the waveform
- `p_conf`: The confidence in the P-wave prediction (higher is better)
- `s_conf`: The confidence in the S-wave prediction (higher is better)
- `p_time_rel`: The P-wave arrival time relative to the first prediction
- `s_time_rel`: The S-wave arrival time relative to the first prediction


## Contributing

Contributions to PhaseHunter are welcome! Feel free to open a pull request with your changes or improvements.

While I am working on a nicer repo, please consider trying the demo here: https://huggingface.co/spaces/crimeacs/phase-hunter
