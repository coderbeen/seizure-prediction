from edflib.epochs import Epochs, ContEpochs

epochs = Epochs(
    "data/chb01/",
    sph=5,
    pil=60,
    psl=120,
    iid=120,
    min_preictal=30,
)

# print(epochs.table)
print(epochs.get_epochs_table())
# print(epochs.get_epochs_table(evenly_divided=False))
# print(epochs.get_epochs_table("Preictal"))
# print(epochs.get_epochs_table("Preictal", evenly_divided=False))
# print(epochs.get_epochs_table("Interictal"))
# print(epochs.get_epochs_table("Interictal", evenly_divided=False))
# print("\n\n")
# print(epochs.get_seizure_names())
# print(epochs.get_seizure_names("Preictal"))
# print(epochs.get_seizure_names("Interictal"))
# print("\n\n")
# pi_seizure = epochs.get_seizure_names("Preictal")[0]
# print(epochs.get_epochs_table(seizures=pi_seizure))
# print(epochs.get_epochs_table(seizures=pi_seizure, evenly_divided=False))
# print(epochs.get_epochs_table(klass="Preictal", seizures=pi_seizure))
# print(
#     epochs.get_epochs_table(klass="Preictal", seizures=pi_seizure, evenly_divided=False)
# )

# epochs = ContEpochs("data/chb08/", sph=0, psl=120)
# print(epochs.table)
# seizure = epochs.get_trainable_seizures()[2]
# print(epochs.get_epochs_table(seizure))
