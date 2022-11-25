import shutil
from typing import Dict, List
import biosignals.dataset as bd
import biosignals.split as bs
import biosignals.features as bf
from random import Random
import pandas as pd
import os

# Full data preparation


# Prepare the cluster assignments
def prepare_clusters() -> pd.DataFrame:
    print('Preparing clusters')
    per_chan, _ = bd.combined_dfs(set(bd.PARTICIPANTS))
    cluster_df = bd.cluster_channels(per_chan, n_clusters=32)
    print(cluster_df)
    if not os.path.exists('prepared'):
        os.mkdir('prepared')
    cluster_df.to_pickle('prepared/clusters.pickle')
    return cluster_df


# Read cluster assignments
def read_clusters() -> pd.DataFrame:
    print('Reading clusters')
    return pd.read_pickle('prepared/clusters.pickle')


# Read cluster assignments (or make and cache them)
def ensure_clusters() -> pd.DataFrame:
    if os.path.exists('prepared/clusters.pickle'):
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
    for r, c in spec.items():
        for i in range(c):
            print(f'Splitting {r.pretty_name()} {i}')
            data_df = splitter.split(marked, r, conf=conf, rand=rand)
            print('... adding features')
            bf.extract_features(data_df, extractors)
            print('... adding cluster info')
            final_df = bd.add_cluster_info(data_df, cluster_df)
            print('... writing to disk')
            dest_path = os.path.join(dest_dir, f'{r.pretty_name()}_{i}.pickle')
            final_df.to_pickle(dest_path)


# An example of how to load and process data
# Tests writing and loading to the 'example' prepared dataset
def prepare_example():
    rand = Random(42)
    conf = bs.DEFAULT_WINDOW_CONFIG
    perm = bs.generate_perm(rand)
    splitter = bs.RandomSplitter(
        perm, {bs.Role.TRAIN: 98, bs.Role.VALIDATE: 1, bs.Role.TEST: 1})
    extractors = bf.default_extractors()

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
class LazyFrame:
    def __init__(self, path: str):
        self._path = path

    def load(self) -> pd.DataFrame:
        return pd.read_pickle(self._path)


# Read prepared data from a directory
def read_prepared(name: str) -> Dict[bs.Role, List[LazyFrame]]:
    dest_dir = os.path.join(f'prepared/{name}')
    assert os.path.isdir(dest_dir)
    fns = sorted(os.listdir(dest_dir))
    d = {}
    for r in (bs.Role.TRAIN, bs.Role.VALIDATE, bs.Role.TEST):
        d[r] = [LazyFrame(os.path.join(dest_dir, fn)) for fn in fns if fn.startswith(r.pretty_name())]
    return d
