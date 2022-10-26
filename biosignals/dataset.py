import os.path
from typing import Tuple
import pynwb
import pandas as pd
import numpy as np

# See https://www.nature.com/articles/s41597-022-01542-9#Sec7 for dataset details.
# Also see the dataset authors' code for processing it:
# https://github.com/neuralinterfacinglab/SingleWordProductionDutch

EEG_SAMPLE_RATE = 1024
AUDIO_SAMPLE_RATE = 48000

DATASET_FILE_FORMAT = 'datasets/SingleWordProductionDutch-iBIDS/sub-{part}/ieeg/sub-{part}_task-wordProduction_{fn}'

PARTICIPANTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

FILES = [
    'channels.tsv',
    'events.tsv',
    'ieeg.json',
    'ieeg.nwb',
    'space-ACPC_coordsystem.json',
    'space-ACPC_electrodes.tsv'
]


# Sanity check that everything is present in the dataset
def check_dataset():
    for part in PARTICIPANTS:
        for fn in FILES:
            full_fn = DATASET_FILE_FORMAT.format(part=part, fn=fn)
            assert os.path.isfile(full_fn), f'missing {full_fn}'


# Read EEG data for a given participant
# Use like:
#     x = read_ieeg('01')
#     y = x.acquisition['iEEG']
#     z = x.acquisition['Stimulus']
# This function is not all that useful because it closes the file before returning.
# Mostly it's just for poking around the container format. To get the data, use
# read_ieeg_data instead
def read_ieeg_container(part: str) -> pynwb.file.NWBFile:
    full_fn = DATASET_FILE_FORMAT.format(part=part, fn='ieeg.nwb')
    with pynwb.NWBHDF5IO(full_fn) as f:
        return f.read()


# Read a tuple of (eeg, stimulus, audio), all Numpy arrays
# eeg has shape (num_channels, num_eeg_samples) and dtype float64
# stimulus has shape (num_eeg_samples,) and dtype string (object)
# audio has shape (num_audio_samples,) and dtype string (object)
# where num_eeg_samples/EEG_SAMPLE_RATE == num_audio_samples/AUDIO_SAMPLE_RATE in seconds
def read_ieeg_data(part: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_fn = DATASET_FILE_FORMAT.format(part=part, fn='ieeg.nwb')
    with pynwb.NWBHDF5IO(full_fn) as f:
        container = f.read()
        eeg = container.acquisition['iEEG'].data[:].transpose()
        eeg_samples = eeg.shape[1]
        stimulus = container.acquisition['Stimulus'].data[:]
        stimulus_samples = stimulus.shape[0]
        assert eeg_samples == stimulus_samples
        audio = container.acquisition['Audio'].data[:]
        audio_samples = audio.shape[0]
        eeg_seconds = float(eeg_samples) / EEG_SAMPLE_RATE
        audio_seconds = float(audio_samples) / AUDIO_SAMPLE_RATE
        assert np.isclose(eeg_seconds, audio_seconds)
        return (eeg, stimulus, audio)


# Read TSV data for a given participant and category
# For channels.tsv only name is interesting? Not sure why this file exists
# I think the name for a given index corresponds to the channel at the given index
# in the eeg array.
# For events.tsv the whole thing is interesting
# For space_space-ACPC_electrodes.tsv x y z are interesting for physical locations
# Match this to channel by looking at index of name in channels.tsv.
# (Electrodes is a superset of channels? Maybe channels just contains only good data.)
def read_tsv(part: str, fn: str) -> pd.DataFrame:
    full_fn = DATASET_FILE_FORMAT.format(part=part, fn=fn)
    return pd.read_csv(full_fn, delimiter='\t')
