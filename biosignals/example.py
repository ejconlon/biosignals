import biosignals.dataset as bd
import biosignals.split as bs
from random import Random


# An example of how to load and process data
def example():
    # Random seed
    rand = Random(42)

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
    train_df = splitter.split(bs.Role.TRAIN, rand=rand)
    validate_df = splitter.split(bs.Role.VALIDATE, rand=rand)
    test_df = splitter.split(bs.Role.TEST, rand=rand)
    # Print just to take a look
    print(train_df)
    print(validate_df)
    print(test_df)

    # Features - TODO!
