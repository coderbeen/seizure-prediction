import numpy as np
from matplotlib import pyplot as plt

from edflib.datastore import DataStore

sdir = "/home/ameer/Documents/dataStores/physionet.org/files/chbmit/1.0.0/chb01/"
ds_kws = dict(
    seg=4,
    sph=0,
    pil=35,
    psl=180,
    iid=60,
    min_preictal=30,
    phop=4,
    ihop=4,
    # max_overlap = 0.25,
    resampling_strategy=None,
)
ds = DataStore(sdir, **ds_kws)

# for seizure in ds.get_trainable_seizures():
#     xtrain, ytrain = ds.get_train_data(seizure)
#     print(f"Xtrain has the shape: {xtrain.shape}")
#     print(f"\tThere is {np.count_nonzero(ytrain==0): 5d} interictal frames.")
#     print(f"\tThere is {np.count_nonzero(ytrain==1): 5d} preictal frames.")

#     xtest, ytest = ds.get_test_data(seizure)
#     print(f"Xtest has the shape: {xtest.shape}")
#     print(f"\tThere is {np.count_nonzero(ytest==0): 5d} interictal frames.")
#     print(f"\tThere is {np.count_nonzero(ytest==1): 5d} preictal frames.")

#     xtest, ytest = ds.get_test_data(seizure, resampled=True)
#     print(f"Xtest has the shape: {xtest.shape}")
#     print(f"\tThere is {np.count_nonzero(ytest==0): 5d} interictal frames.")
#     print(f"\tThere is {np.count_nonzero(ytest==1): 5d} preictal frames.")

#     print("\n")

ds.load_data()
for seizure in ds.get_trainable_seizures():

    x, y = ds.get_test_data(seizure)
    print(f"Subject has {x.shape}.")
    print(f"Subject has {np.count_nonzero(y==0): 5d} interictal frames.")
    print(f"Subject has {np.count_nonzero(y==1): 5d} preictal frames.")
    print(f"Data has a mean of {np.mean(x)} and a std of {np.std(x)}")
    print("\n")
