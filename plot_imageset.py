import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

data = pickle.load(open("denoising_psnr_imageset.p", "rb"))

psnrs = []
idxs = []

for file, (idx, psnr) in data.items():
    psnrs.append(psnr)
    idxs.append(idx)

x = np.arange(len(psnrs))
plt.plot(x, psnrs)

plt.legend(["PSNR"], loc='best')
plt.legend(fontsize="medium")

plt.savefig('psnr_validate.png')
