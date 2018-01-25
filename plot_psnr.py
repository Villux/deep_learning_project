import pickle
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

psrn_histories = []

plt.figure(figsize=(15, 10))

for file in glob.glob("*.p"):
    data = pickle.load(open(file, "rb"))
    psrn_histories.append(data["psnr_history"])

for line in psrn_histories:
    x = np.arange(len(line))
    plt.plot(x, line)

plt.legend(['denoising', 'denoising-simple', 'denoising-no-skip', 'denoising-large-skip',
            'denoising-inc-no-skip', 'denoising-inc-dec-filter-size'], loc='best')
plt.legend(fontsize="medium")

plt.savefig('psnr_history.png')
