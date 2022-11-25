import shutil
from typing import Dict, List
from pandas.testing import assert_frame_equal
import biosignals.dataset as bd
import biosignals.split as bs
import biosignals.features as bf
from random import Random
import pandas as pd
import os

# Full data preparation


# An example of how to load and process data
# Tests writing and loading to the 'example' prepared dataset
def prepare_example():
    # Random seed
    rand = Random(42)

    # Window config
    conf = bs.DEFAULT_WINDOW_CONFIG

    # Clustering channels
    print('Clustering channels')
    per_chan, _ = bd.combined_dfs(set(bd.PARTICIPANTS))
    cluster_df = bd.cluster_channels(per_chan, n_clusters=32)
    print(cluster_df)

    # Reading splits

    # Read marked data
    print('Reading data')
    marked = bd.read_marked_data()
    # Generate permutation
    perm = bs.generate_perm(rand)
    # Define splits
    rand_splitter = bs.RandomSplitter(
        marked, {bs.Role.TRAIN: 80, bs.Role.VALIDATE: 10, bs.Role.TEST: 10}, perm)
    # Split the data
    print('Splitting data')
    # TODO Split train and validate sets too
    # train_df = splitter.split(bs.Role.TRAIN, conf=conf, rand=rand)
    # validate_df = splitter.split(bs.Role.VALIDATE, conf=conf, rand=rand)
    rand_test_df = rand_splitter.split(bs.Role.TEST, conf=conf, rand=rand)
    # Print just to take a look
    # print(train_df)
    # print(validate_df)
    print(rand_test_df)

    # Variant with per-participant splits - just an example
    part_splitter = bs.PartSplitter(marked, {bs.Role.TEST: set(['01', '02'])})
    part_test_df = part_splitter.split(bs.Role.TEST, conf=conf, rand=rand)
    print(part_test_df)

    # Features - TODO add more features
    extractors = bf.default_extractors()
    test_feat_df = bf.extract_features(rand_test_df, extractors)
    print(test_feat_df)

    # Add cluster/spatial info
    final_feat_df = bd.add_cluster_info(cluster_df, test_feat_df)
    print(final_feat_df)

    # TODO Write prepared data to dist
    prep = {bs.Role.TEST: [final_feat_df]}
    write_prepared('example', prep)

    # Check that we can read what we write
    prep_read = read_prepared('example')
    assert len(prep_read) == 3
    assert len(prep_read[bs.Role.TRAIN]) == 0
    assert len(prep_read[bs.Role.VALIDATE]) == 0
    assert len(prep_read[bs.Role.TEST]) == 1
    check_df = prep_read[bs.Role.TEST][0].load()
    assert_frame_equal(final_feat_df, check_df)


# Write prepared data to a directory
def write_prepared(name: str, role_dfs: Dict[bs.Role, List[pd.DataFrame]]):
    if not os.path.exists('prepared'):
        os.mkdir('prepared')
    dest_dir = os.path.join(f'prepared/{name}')
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)
    for r, dfs in role_dfs.items():
        for i, df in enumerate(dfs):
            dest_path = os.path.join(dest_dir, f'{r.pretty_name()}_{i}.pickle')
            df.to_pickle(dest_path)


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
