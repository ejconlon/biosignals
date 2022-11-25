import biosignals.dataset as bd
import biosignals.split as bs
import biosignals.features as bf
from random import Random

# Full data preparation


# An example of how to load and process data
def prepare():
    # Random seed
    rand = Random(42)

    # Window config
    conf = bs.DEFAULT_WINDOW_CONFIG

    # Clustering channels
    print('Clustering channels')
    per_chan, _ = bd.combined_dfs(set(bd.PARTICIPANTS))
    clust_df = bd.cluster_channels(per_chan, n_clusters=32)
    print(clust_df)

    # Reading splits

    # Read marked data
    print('Reading data')
    marked = bd.read_marked_data()
    # Generate permutation
    perm = bs.generate_perm(rand)
    # Define splits
    splitter = bs.RandomSplitter(
        marked, {bs.Role.TRAIN: 80, bs.Role.VALIDATE: 10, bs.Role.TEST: 10}, perm)
    # Split the data
    print('Splitting data')
    # TODO Split train and validate sets too
    # train_df = splitter.split(bs.Role.TRAIN, conf=conf, rand=rand)
    # validate_df = splitter.split(bs.Role.VALIDATE, conf=conf, rand=rand)
    test_df = splitter.split(bs.Role.TEST, conf=conf, rand=rand)
    # Print just to take a look
    # print(train_df)
    # print(validate_df)
    print(test_df)

    # Features - TODO add more features
    extractors = bf.default_extractors()
    test_feat_df = bf.extract_features(test_df, extractors)
    print(test_feat_df)

    # TODO Write prepared data to dist
