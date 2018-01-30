import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', '-m', type=str, default='basic', help="Model name")
args = argparser.parse_args()

data = pickle.load(open(f'denoising_psnr_imageset_{args.model}.p, "rb"))

psnrs = []
idxs = []

for file, (idx, psnr) in data.items():
    psnrs.append(psnr)
    idxs.append(idx)

print(f"AVG: {np.mean(psnrs)}")
print(f"MAX: {np.max(psnrs)}")
print(f"MIN: {np.min(psnrs)}")

x = np.arange(len(psnrs))
plt.plot(x, psnrs)

plt.legend(["PSNR"], loc='best')
plt.legend(fontsize="medium")
plt.xlabel("Image")
plt.ylabel("Best PSNR")
plt.savefig('psnr_validate.png')

plt.clf()

plt.scatter(idxs, psnrs)
plt.xlabel("Best PSNR at iteration")
plt.ylabel("Best PSNR")
plt.savefig('psnr_validate_scatter.png')
