from matplotlib import pyplot as plt

from edflib.epochs import Epochs
from edflib.datastore import DataStore
from edflib import plot as edfplt

sdir = "/home/ameer/Documents/dataStores/physionet.org/files/chbmit/1.0.0/chb08/"
# ds = DataStore(sdir, 8, 5, 60, 120, 120)

# # edfplt.timeline(ds)

# seizure = ds.get_trainable_seizures()[0]

# # epochs = ds.get_epochs_table("Interictal", seizure)
# # file, *_ = epochs.iloc[0]
# # data = ds.read_file(file)
# # plt.figure(f"Seizure {seizure} / File {file}", layout="constrained")
# # edfplt.plot(data)

# epochs = ds.get_epochs_table("Preictal", seizure)
# file, *_ = epochs.iloc[0]
# data = ds.read_file(file, (0, 60))
# plt.figure(f"Seizure {seizure} / File {file}", layout="constrained")
# edfplt.plot(data)

# plt.show()

epochs = Epochs(sdir, 0, 35, 120, 120)
print(epochs.seizures)
print(epochs.get_epochs_table())
edfplt.timeline(epochs)
# edfplt.summarize(epochs, pil=30, psl=120, iid=120)
plt.show()
