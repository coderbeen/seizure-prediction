from edflib.summary import Summary

# sdir = "/home/ameer/Documents/dataStores/physionet.org/files/chbmit/1.0.0/chb01/"
# summary = Summary(sdir)


summary = Summary("data/chb08/")

print(summary.records)

print(summary.get_preictal_epochs(sph=5, pil=60, psl=120))

print(summary.get_interictal_epochs(iid=120))

print(summary.seizures)

print(summary.pretty_records()[0:3])
