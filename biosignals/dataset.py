import os
from typing import Dict, List, Set, Tuple
import pynwb
import pandas as pd
import numpy as np
import soundfile
import aifc

# See https://www.nature.com/articles/s41597-022-01542-9#Sec7 for dataset details.
# Also see the dataset authors' code for processing it:
# https://github.com/neuralinterfacinglab/SingleWordProductionDutch

EEG_SAMPLE_RATE = 1024
AUDIO_SAMPLE_RATE = 48000
EEG_LOW_CUTOFF_HZ = 0
EEG_HIGH_CUTOFF_HZ = 512

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
        assert np.isclose(eeg_seconds, audio_seconds, atol=0.005)
        return (eeg, stimulus, audio)


# Read a (eeg, stimulus, audio) for all participants
def real_all_ieeg_data() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    all_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for part in PARTICIPANTS:
        tup = read_ieeg_data(part)
        all_data[part] = tup
    return all_data


# Read TSV data for a given participant and category
# For channels.tsv only name is interesting? Not sure why this file exists
# I think the name for a given index corresponds to the channel at the given index
# in the eeg array.
# For events.tsv the whole thing is interesting
# For space-ACPC_electrodes.tsv x y z are interesting for physical locations
# Match this to channel by looking at index of name in channels.tsv.
# (Electrodes is a superset of channels? Maybe channels just contains only good data.)
def read_tsv(part: str, fn: str) -> pd.DataFrame:
    full_fn = DATASET_FILE_FORMAT.format(part=part, fn=fn)
    return pd.read_csv(full_fn, delimiter='\t')


# Returns per-channel and per-part dataframes, each labeled with part column
def read_part_dfs(part: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    chans = read_tsv(part, 'channels.tsv')
    chans = chans[['name']]
    chans['part'] = part
    chans['channel_id'] = chans.index
    electrodes = read_tsv(part, 'space-ACPC_electrodes.tsv')
    electrodes = electrodes[['name', 'x', 'y', 'z']]
    per_chan = chans.merge(electrodes, on='name', how='inner')
    assert len(per_chan) == len(chans)
    per_part = read_tsv(part, 'events.tsv')
    per_part['part'] = part
    return (per_chan, per_part)


# Returns per-channel and per-part dataframes for the given set of participants
def combined_dfs(parts: Set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert len(parts) > 0
    pairs = [read_part_dfs(part) for part in sorted(parts)]
    combined_per_chan = pd.concat((p[0] for p in pairs), ignore_index=True)
    combined_per_part = pd.concat((p[1] for p in pairs), ignore_index=True)
    return (combined_per_chan, combined_per_part)


# Write audio from the given part to files in the directory
# Writes marked and unmarked versions
def write_part_aiff(part: str, dirname: str):
    assert os.path.isdir(dirname)
    unmarked_fn = f'{dirname}/part_{part}_unmarked.aiff'
    marked_fn = f'{dirname}/part_{part}_marked.aiff'
    assert not os.path.exists(unmarked_fn)
    assert not os.path.exists(marked_fn)
    (_, stim, audio) = read_ieeg_data(part)
    stim_ixs = [ix for ix in np.where(np.roll(stim, 1) != stim)[0]]
    soundfile.write(unmarked_fn, audio, AUDIO_SAMPLE_RATE, format='aiff')
    with aifc.open(unmarked_fn, 'r') as f:
        params = f.getparams()
        nframes = f.getnframes()
        data = f.readframes(nframes)
    with aifc.open(marked_fn, 'w') as f:
        f.aiff()
        f.setparams(params)
        f.writeframes(data)
        next_id = 1
        for num, esamp in enumerate(stim_ixs):
            asamp = int(float(esamp) / EEG_SAMPLE_RATE * AUDIO_SAMPLE_RATE)
            val = stim[esamp]
            is_prompt = len(val) > 0
            # The marks are either s{num} for start or e{num} for end
            # The onset marks should be o{num} for onset
            mark_val = f's{num // 2}' if is_prompt else f'e{num // 2}'
            f.setmark(next_id, asamp, mark_val.encode())
            next_id += 1
            # Write onset mark if this is the start of a window
            if is_prompt:
                # This is very dumb - just put a mark 25% of the way into a window
                # Replace this with onset detection?
                next_asamp = int(float(stim_ixs[num + 1])) / EEG_SAMPLE_RATE * AUDIO_SAMPLE_RATE
                onset_asamp = int(asamp + 0.25 * (next_asamp - asamp))
                onset_mark_val = f'o{num // 2}'
                f.setmark(next_id, onset_asamp, onset_mark_val.encode())
                next_id += 1


# Write audio from all parts to the given directory
def write_all_aiff(dirname: str):
    assert not os.path.exists(dirname)
    os.mkdir(dirname)
    for part in PARTICIPANTS:
        write_part_aiff(part, dirname)


# Read marks from the audio file and return onsets (at the eeg sample rate, like stim)
# The 'name' parameter is the variant of the aiff file (like "marked") to read.
# This code is horrible because every audio editor handles marks horribly.
# Sometimes they're duplicated, sometimes they're cycled, ugh.
def read_part_onsets(part: str, dirname: str, name: str) -> np.ndarray:
    assert os.path.isdir(dirname)
    fn = f'{dirname}/part_{part}_{name}.aiff'
    with aifc.open(fn, 'r') as f:
        marks = f.getmarkers()
    assert marks is not None
    onsets: List[int] = []
    window = 0
    seeking = 's'
    last_num = -1
    last_val = 'xxx'.encode()
    for mark in marks:
        (num, asamp, mark_val) = mark
        if mark_val == last_val:
            continue
        elif num < last_num:
            break
        expected_val = f'{seeking}{window}'.encode()
        assert mark_val == expected_val
        if seeking == 's':
            seeking = 'o'
        elif seeking == 'o':
            esamp = int(float(asamp) / AUDIO_SAMPLE_RATE * EEG_SAMPLE_RATE)
            onsets.append(esamp)
            seeking = 'e'
        elif seeking == 'e':
            window += 1
            seeking = 's'
        else:
            assert False, 'invalid'
        last_num = num
        last_val = mark_val
    # Check outside of this function that this matches the number of stimuli / 2
    # assert len(onsets) == 100
    return np.array(onsets)


# Read all onsets from aiff files in a given directory.
# The 'name' parameter is the variant of the aiff file (like "marked") to read.
def read_all_onsets(dirname: str, name: str) -> Dict[str, np.ndarray]:
    assert os.path.isdir(dirname)
    all_onsets: Dict[str, np.ndarray] = {}
    for part in PARTICIPANTS:
        onsets = read_part_onsets(part, dirname, name)
        all_onsets[part] = onsets
    return all_onsets
