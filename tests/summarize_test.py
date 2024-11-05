from matplotlib import pyplot as plt

from edflib.summary import Summary
from edflib import plot as edfplt

sdir = "/home/ameer/Documents/dataStores/physionet.org/files/chbmit/1.0.0/chb01/"

summary = Summary(sdir)
edfplt.summarize(summary, pil=60, psl=120, iid=180)
plt.show()
