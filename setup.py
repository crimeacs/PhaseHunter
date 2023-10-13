from setuptools import setup, find_packages

setup(
    name='PhaseHunter',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
        'obspy',
        'pytorch-lightning',
        'lightning',
        'wandb',
    ],
    author='Artemii Novoselov',
    author_email='anovosel@stanford.edu',
    description='PhaseHunter is a state-of-the-art deep learning model for precise estimation and uncertainty quantification of seismic phase onset times.',
    url='https://github.com/crimeacs/PhaseHunter',
)