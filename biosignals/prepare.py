from dataclasses import dataclass, replace
import shutil
from typing import Dict, List, Optional
import biosignals.dataset as bd
import biosignals.split as bs
import biosignals.features as bf
from random import Random
import pandas as pd
import os

# Full data preparation

# Prep names
HOLDOUT_PREP_NAMES = [f'holdout_{p}' for p in bd.PARTICIPANTS]
ONLINE_PREP_NAMES = [f'online_{p}' for p in bd.PARTICIPANTS]
STANDARD_PREP_NAMES = ['rand']
# Comment this out if you don't want to train holdouts by default
STANDARD_PREP_NAMES.extend(HOLDOUT_PREP_NAMES)


# Number of clusters - if this changes, you have to regen all data
NUM_CLUSTERS = 32


# Number of ms for overlapping windows
ONLINE_STEP_MS = 50


# EEG_SAMPS_PER_MS = 1024 / 1000
# Since samps/ms is basically 1, we just use round numbers here
DEFAULT_WINDOW_CONFIG = bs.WindowConfig(
    pre_len=500,
    post_len=250,
    max_jitter=0,
    exclude_len=500
)


# This comes from https://mne.tools/dev/auto_examples/time_frequency/time_frequency_global_field_power.html
FREQ_BANDS = [
    bf.Band('delta_power', 1, 3),
    bf.Band('theta_power', 4, 7),
    bf.Band('alpha_power', 8, 12),
    bf.Band('beta_power', 13, 25),
    bf.Band('gamma_power', 30, 45)
]

# Number of segments for band power
NUM_SEGMENTS = 1

# Feature extractors for band power
FREQ_EXTRACTORS = [bf.SegmentBandPowerExtractor(NUM_SEGMENTS, FREQ_BANDS)]


# Feature extractors to use by default
def default_extractors() -> List[bf.Extractor]:
    return list(FREQ_EXTRACTORS)


# Prepare the cluster assignments
def prepare_clusters() -> pd.DataFrame:
    print('Preparing clusters')
    per_chan, _ = bd.combined_dfs(set(bd.PARTICIPANTS))
    cluster_df = bd.cluster_channels(per_chan, n_clusters=NUM_CLUSTERS)
    print(cluster_df)
    if not os.path.exists('prepared'):
        os.mkdir('prepared')
    cluster_df.to_parquet('prepared/clusters.parquet')
    return cluster_df


# Read cluster assignments
def read_clusters() -> pd.DataFrame:
    print('Reading clusters')
    return pd.read_parquet('prepared/clusters.parquet', use_nullable_dtypes=False)


# Read cluster assignments (or make and cache them)
def ensure_clusters() -> pd.DataFrame:
    if os.path.exists('prepared/clusters.parquet'):
        return read_clusters()
    else:
        return prepare_clusters()


# Prepare and write data
def prepare_splits(
    name: str,
    conf: bs.WindowConfig,
    rand: Random,
    splitter: bs.Splitter,
    extractors: List[bf.Extractor],
    spec: Dict[bs.Role, int]
):
    cluster_df = ensure_clusters()

    print('Reading data')
    marked = bd.read_marked_data()

    print('Cleaning destination')
    if not os.path.exists('prepared'):
        os.mkdir('prepared')
    dest_dir = os.path.join(f'prepared/{name}')
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)

    print('Preparing splits')
    for r in (bs.Role.TRAIN, bs.Role.VALIDATE, bs.Role.TEST):
        c = spec.get(r)
        if c is not None:
            for i in range(c):
                print(f'Splitting {r.pretty_name()} {i}')
                data_df = splitter.split(marked, r, conf=conf, rand=rand)
                print('... adding features')
                bf.extract_features(data_df, extractors)
                print('... adding cluster info')
                final_df = bd.add_cluster_info(data_df, cluster_df)
                print('... writing to disk')
                dest_path = os.path.join(dest_dir, f'{r.pretty_name()}_{i}.parquet')
                final_df.to_parquet(dest_path)


# An example of how to load and process data
# Tests writing and loading to the 'example' prepared dataset
def prepare_example():
    rand = Random(42)
    conf = DEFAULT_WINDOW_CONFIG
    perms = bs.generate_perms(bd.PARTICIPANTS, rand)
    splitter = bs.RandomSplitter(
        perms, {bs.Role.TRAIN: 98, bs.Role.VALIDATE: 1, bs.Role.TEST: 1})
    extractors = default_extractors()

    prepare_splits(
        name='example',
        conf=conf,
        rand=rand,
        splitter=splitter,
        extractors=extractors,
        spec={bs.Role.TEST: 1}
    )

    print('Testing example read')
    prep_read = read_prepared('example')
    assert len(prep_read) == 3
    assert len(prep_read[bs.Role.TRAIN]) == 0
    assert len(prep_read[bs.Role.VALIDATE]) == 0
    assert len(prep_read[bs.Role.TEST]) == 1
    check_df = prep_read[bs.Role.TEST][0].load()
    assert len(check_df) > 0
    print(check_df)


# Prepare a randomly-split set
def prepare_rand():
    rand = Random(42)
    conf = DEFAULT_WINDOW_CONFIG
    perms = bs.generate_perms(bd.PARTICIPANTS, rand)
    splitter = bs.RandomSplitter(
        perms, {bs.Role.TRAIN: 90, bs.Role.VALIDATE: 0, bs.Role.TEST: 10})
    extractors = default_extractors()

    prepare_splits(
        name='rand',
        conf=conf,
        rand=rand,
        splitter=splitter,
        extractors=extractors,
        spec={
            bs.Role.TRAIN: 1,
            bs.Role.VALIDATE: 1,
            bs.Role.TEST: 1
        }
    )


# Ensure that the rand dataset exists
def ensure_rand():
    if not has_prepared('rand'):
        prepare_rand()


# Prepare a jittered set to pump up the training examples
def prepare_jit():
    rand = Random(42)
    conf = replace(DEFAULT_WINDOW_CONFIG, max_jitter=50)
    perms = bs.generate_perms(bd.PARTICIPANTS, rand)
    splitter = bs.RandomSplitter(
        perms, {bs.Role.TRAIN: 80, bs.Role.VALIDATE: 0, bs.Role.TEST: 20})
    extractors = default_extractors()

    prepare_splits(
        name='jit',
        conf=conf,
        rand=rand,
        splitter=splitter,
        extractors=extractors,
        spec={
            bs.Role.TRAIN: 2,
            bs.Role.VALIDATE: 0,
            bs.Role.TEST: 1
        }
    )


# Prepare holdout set for given participant
# This is just that participant in a validation set.
def prepare_holdout_part(part: str):
    assert part in bd.PARTICIPANTS

    rand = Random(42)
    extractors = default_extractors()

    splitter = bs.PartSplitter({bs.Role.VALIDATE: set([part])})

    prepare_splits(
        name=f'holdout_part_{part}',
        conf=DEFAULT_WINDOW_CONFIG,
        rand=rand,
        splitter=splitter,
        extractors=extractors,
        spec={
            bs.Role.TRAIN: 0,
            bs.Role.VALIDATE: 1,
            bs.Role.TEST: 0
        }
    )


# Ensure holdout set for given participant
def ensure_holdout_part(part: str):
    if not has_prepared(f'holdout_part_{part}'):
        prepare_holdout_part(part)


# Ensure all holdout sets
def ensure_holdout():
    for part in bd.PARTICIPANTS:
        ensure_holdout_part(part)


# Ensure online set for given participant
# This is like the holdout set but with overlapping windows and labels -1.
def prepare_online_part(step_ms: int, part: str):
    assert part in bd.PARTICIPANTS

    rand = Random(42)
    extractors = default_extractors()

    splitter = bs.OnlineSplitter(step_ms, {bs.Role.VALIDATE: set([part])})

    prepare_splits(
        name=f'online_part_{step_ms}_{part}',
        conf=DEFAULT_WINDOW_CONFIG,
        rand=rand,
        splitter=splitter,
        extractors=extractors,
        spec={
            bs.Role.TRAIN: 0,
            bs.Role.VALIDATE: 1,
            bs.Role.TEST: 0
        }
    )


# Ensure online set for given participant
def ensure_online_part(step_ms: int, part: str):
    if not has_prepared(f'online_part_{step_ms}_{part}'):
        prepare_online_part(step_ms, part)


# Same thing
def ensure_online(step_ms: int):
    for part in bd.PARTICIPANTS:
        ensure_online_part(step_ms, part)


# Loads dataframes from files/memory
class FrameLoader:
    def load(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        raise NotImplementedError()


# Loads a dataframe from a Parquet file
@dataclass(frozen=True)
class ParquetFrameLoader(FrameLoader):
    path: str

    def load(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        return pd.read_parquet(self.path, columns=columns, use_nullable_dtypes=False)


# "Loads" a dataframe from memory
@dataclass(frozen=True)
class MemoryFrameLoader(FrameLoader):
    frame: pd.DataFrame

    def load(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            return self.frame
        else:
            return self.frame[columns]


# Read prepared data from a directory (smart)
def read_prepared(name: str) -> Dict[bs.Role, List[FrameLoader]]:
    xs = name.split('_')
    category = xs[0]
    if (len(xs) == 2 and xs[0] == 'holdout') or (len(xs) == 3 and xs[0] == 'online'):
        part = xs[1] if category == 'holdout' else xs[2]
        category = 'holdout_part' if xs[0] == 'holdout' else f'online_part_{xs[1]}'
        train_loaders = []
        test_loaders = []
        for p in bd.PARTICIPANTS:
            part_name = f'{category}_{p}'
            d = read_prepared_raw(part_name)
            if p == part:
                test_loaders.extend(d[bs.Role.VALIDATE])
            else:
                train_loaders.extend(d[bs.Role.VALIDATE])
        return {
            bs.Role.TRAIN: train_loaders,
            bs.Role.VALIDATE: [],
            bs.Role.TEST: test_loaders
        }
    else:
        return read_prepared_raw(name)


# Read prepared data from a directory (raw)
def read_prepared_raw(name: str) -> Dict[bs.Role, List[FrameLoader]]:
    dest_dir = os.path.join(f'prepared/{name}')
    assert os.path.isdir(dest_dir)
    fns = sorted(os.listdir(dest_dir))
    d: Dict[bs.Role, List[FrameLoader]] = {}
    for r in (bs.Role.TRAIN, bs.Role.VALIDATE, bs.Role.TEST):
        d[r] = [
            ParquetFrameLoader(os.path.join(dest_dir, fn))
            for fn in fns if fn.startswith(r.pretty_name())
        ]
    return d


# Check if there is prepared data under the given name (smart)
def has_prepared(name: str) -> bool:
    xs = name.split('_')
    if (len(xs) == 2 and xs[0] == 'holdout') or (len(xs) == 3 and xs[0] == 'online'):
        category = 'holdout_part' if xs[0] == 'holdout' else f'online_part_{xs[1]}'
        return all(has_prepared_raw(f'{category}_{p}') for p in bd.PARTICIPANTS)
    else:
        return has_prepared_raw(name)


# Check if there is prepared data under the given name (raw)
def has_prepared_raw(name: str) -> bool:
    dest_dir = os.path.join(f'prepared/{name}')
    return os.path.isdir(dest_dir)


# Ensure all of the important prepared datasets
def ensure_all():
    ensure_rand()
    ensure_holdout()
    # Beware: online takes 10G and a few hours to generate!
    # ensure_online(ONLINE_STEP_MS)
