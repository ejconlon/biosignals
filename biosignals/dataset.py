import os.path
import pynwb

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
def read_ieeg(part: str) -> pynwb.file.NWBFile:
    fn = DATASET_FILE_FORMAT.format(part=part, fn='ieeg.nwb')
    with pynwb.NWBHDF5IO(fn) as f:
        return f.read()
