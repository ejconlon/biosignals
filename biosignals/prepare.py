from dataclasses import dataclass
import shutil
from typing import Dict, List, Optional
import biosignals.dataset as bd
import biosignals.split as bs
import biosignals.features as bf
from random import Random
import pandas as pd
import os

# Full data preparation


# EEG_SAMPS_PER_MS = 1024 / 1000
# Since samps/ms is basically 1, we just use round numbers here
DEFAULT_WINDOW_CONFIG = bs.WindowConfig(
    pre_len=500,
    post_len=250,
    max_jitter=50,
    exclude_len=500
)


# This comes from https://mne.tools/dev/auto_examples/time_frequency/time_frequency_global_field_power.html
FREQ_BANDS = [
    ('theta_power', 4, 7),
    ('alpha_power', 8, 12),
    ('beta_power', 13, 25),
    ('gamma_power', 30, 45)
]


# Feature extractors for band power
FREQ_EXTRACTORS = [bf.band_power_extractor(name, lo, hi) for (name, lo, hi) in FREQ_BANDS]


# Feature extractors to use by default
def default_extractors() -> List[bf.Extractor]:
    return list(FREQ_EXTRACTORS)


# Prepare the cluster assignments
def prepare_clusters() -> pd.DataFrame:
    print('Preparing clusters')
    per_chan, _ = bd.combined_dfs(set(bd.PARTICIPANTS))
    cluster_df = bd.cluster_channels(per_chan, n_clusters=32)
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
    # Go in this order so feature scaling works when it's added
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
    perm = bs.generate_perm(rand)
    splitter = bs.RandomSplitter(
        perm, {bs.Role.TRAIN: 98, bs.Role.VALIDATE: 1, bs.Role.TEST: 1})
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
    conf = bs.DEFAULT_WINDOW_CONFIG
    perm = bs.generate_perm(rand)
    splitter = bs.RandomSplitter(
        perm, {bs.Role.TRAIN: 80, bs.Role.VALIDATE: 10, bs.Role.TEST: 10})
    extractors = bf.default_extractors()

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


# A simple proxy to lazy-load dataframes
@dataclass(frozen=True)
class LazyFrame:
    path: str

    def load(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        return pd.read_parquet(self.path, columns=columns, use_nullable_dtypes=False)


# Read prepared data from a directory
def read_prepared(name: str) -> Dict[bs.Role, List[LazyFrame]]:
    dest_dir = os.path.join(f'prepared/{name}')
    assert os.path.isdir(dest_dir)
    fns = sorted(os.listdir(dest_dir))
    d = {}
    for r in (bs.Role.TRAIN, bs.Role.VALIDATE, bs.Role.TEST):
        d[r] = [LazyFrame(os.path.join(dest_dir, fn)) for fn in fns if fn.startswith(r.pretty_name())]
    return d
