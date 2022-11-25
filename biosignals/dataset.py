import os
from typing import Dict, List, Set, Tuple
import pynwb
import pandas as pd
import numpy as np
import soundfile
import aifc
import librosa
from sklearn.cluster import BisectingKMeans
from dataclasses import dataclass

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


# NOTE(ejconlon) This is just here to quickly observe clustering in the repl
# Don't use it otherwise.
def easy_cluster_channels(n_clusters=32) -> pd.DataFrame:
    x, _ = combined_dfs(set(PARTICIPANTS))
    return cluster_channels(x, n_clusters=n_clusters)


# Cluster channels to find closest to each cluster centroid
# Takes as input the combined per_chan dataframe (from combined_dfs)
# Input:
# combined_chan_df: DataFrame with x, y, z, part, channel_id
#   for each (part, channel_id)
# n_clusters: number of clusters - should be < 64 (min number of channels)
# Output:
# DataFrame with x, y, z, part, channel_id, cluster_id
#   For each (part, cluster_id) with cluster_id >= 0,
#   maps (part, cluster_id) -> channel_id .
#   If cluster_id < 0 then it is not considered to be in a cluster.
def cluster_channels(combined_chan_df: pd.DataFrame, n_clusters=32) -> pd.DataFrame:
    x = combined_chan_df
    parts = sorted(set(x.part))
    k = BisectingKMeans(n_clusters=n_clusters).fit(x[['x', 'y', 'z']])
    out_map = {}
    for part in parts:
        y = x[x.part == part]
        v = k.transform(y[['x', 'y', 'z']])
        closest_ranked = np.argsort(v.transpose())
        assert closest_ranked.shape == (n_clusters, len(y))
        for cluster_id in range(n_clusters):
            found = False
            for channel_id in closest_ranked[cluster_id]:
                pair = (part, channel_id)
                if pair in out_map:
                    pass
                else:
                    out_map[pair] = cluster_id
                    found = True
                    break
            assert found
    # Assign cluster_id
    cluster_ids = []
    for _, row in x.iterrows():
        pair = (row['part'], row['channel_id'])  # type: ignore
        cluster_id = out_map.get(pair, -1)
        cluster_ids.append(cluster_id)
    z = x.copy(deep=True)
    z.insert(0, 'cluster_id', pd.Series(cluster_ids, index=z.index, dtype=np.int_))
    # Sanity checks
    # Assert there is a cluster entry for each participant
    for cluster_id in range(n_clusters):
        y = z[z.cluster_id == cluster_id]
        assert len(y) == len(parts)
    return z


# Add cluster information to the data dataframe (typically features)
# Inputs:
# data_df: dataframe with 'channel_id', 'part', feature columns
# cluster_df: dataframe with 'cluster_id', 'channel_id', 'part', 'x', 'y', 'z'
# Side effect:
# Returns data_df with columns 'cluster_id', 'x', 'y', 'z' added
def add_cluster_info(data_df: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
    right_df = cluster_df[['part', 'channel_id', 'cluster_id', 'x', 'y', 'z']]
    return pd.merge(data_df, right_df, on=('part', 'channel_id'), how='left', copy=False)


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


# Only read eeg data
def read_ieeg_data_only(part: str) -> np.ndarray:
    full_fn = DATASET_FILE_FORMAT.format(part=part, fn='ieeg.nwb')
    with pynwb.NWBHDF5IO(full_fn) as f:
        container = f.read()
        return container.acquisition['iEEG'].data[:].transpose()


# Read a (eeg, stimulus, audio) for all participants
def read_all_ieeg_data() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    all_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for part in PARTICIPANTS:
        tup = read_ieeg_data(part)
        all_data[part] = tup
    return all_data


# Only read eeg data for all participants
def read_all_ieeg_data_only() -> Dict[str, np.ndarray]:
    all_data: Dict[str, np.ndarray] = {}
    for part in PARTICIPANTS:
        eeg = read_ieeg_data_only(part)
        all_data[part] = eeg
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
    # problematic_markers = []
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

                next_asamp = int(float(stim_ixs[num + 1])) / EEG_SAMPLE_RATE * AUDIO_SAMPLE_RATE
                onset_asamp = 0
                curr_audio = audio[int(asamp):int(next_asamp)]

                audio_rms = librosa.feature.rms(y=curr_audio)[0]
                audio_times = librosa.frames_to_time(np.arange(len(audio_rms)), sr=AUDIO_SAMPLE_RATE)

                r_normalized = (audio_rms - 0.02) / np.std(audio_rms)
                p = np.exp(r_normalized) / (1 + np.exp(r_normalized))

                transition = librosa.sequence.transition_loop(2, [0.5, 0.6])
                full_p = np.vstack([1 - p, p])

                states = librosa.sequence.viterbi_discriminative(full_p, transition).astype(np.int8)

                # Add padding before the detected speech envelope and pass each envelope thru onset detection function
                state_changes = states - np.roll(states, 1)
                # 1 indicates the start of a detected envelope and -1 indicates the end

                # Case: no state changes detected -> need to manually set markers
                if np.count_nonzero(state_changes) == 0:
                    # This is very dumb - just put a mark 25% of the way into a window
                    onset_asamp = int(asamp + 0.25 * (next_asamp - asamp))
                    # problematic_markers.append(num)
                # Case: at least 1 state change detected
                # -----> get start and end of state changes (set as envelope) and do onset detection
                else:
                    # Discard all starts and stops in the middle (assume 1 envelope per word)
                    env_start = np.argwhere(state_changes == 1)[0]
                    env_end = np.argwhere(state_changes == -1)[-1]

                    # Case: no end -> set env_end to the last index
                    if env_end < env_start:
                        env_end = len(state_changes) - 1

                    padding = 4800  # number of samples to pad in front of detected envelope (=0.1s)

                    # Start and end audio sample numbers
                    curr_audio_start = int(audio_times[env_start] * AUDIO_SAMPLE_RATE)
                    if curr_audio_start >= padding:
                        curr_audio_start -= padding
                    curr_audio_end = int(audio_times[env_end] * AUDIO_SAMPLE_RATE)
                    onset_asamp = librosa.onset.onset_detect(y=audio[curr_audio_start:curr_audio_end],
                                                             sr=AUDIO_SAMPLE_RATE,
                                                             units='samples')[0] + curr_audio_start + asamp
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
        # assert mark_val == expected_val # This assertion fails for participants #3, #7, and #10
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


# Read onsets from what we've manually marked
def read_manual_onsets() -> Dict[str, np.ndarray]:
    return read_all_onsets('annotated_markers', 'marked')


# A pair of onsets and eeg data
# Later we can extract windows from this data as we see fit
@dataclass(frozen=True)
class MarkedData:
    onsets: List[int]
    eeg: np.ndarray


# Read marked data for all participants
def read_marked_data() -> Dict[str, MarkedData]:
    part_onsets = read_manual_onsets()
    part_eegs = read_all_ieeg_data_only()
    out = {}
    for part, onsets in part_onsets.items():
        out[part] = MarkedData(list(onsets), part_eegs[part])
    return out
