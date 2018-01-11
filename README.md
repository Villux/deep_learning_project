# deep_learning_project
Repository for Deep Learning course's project work

### How to run
- Python 3.6
- Install anaconda & Jupyter Notebook
- Install conda env with `conda env create -f environment.yml`

### Paniikki computers
- Load anaconda `module load anaconda3`
- Use virtual env `source activate image`

### SSH tunnel from Paniikki computers
In paniikki computer:
- `ipython notebook --no-browser --port=8889`
In Kosh computer:
- `ssh -N -f -L localhost:8889:localhost:8889 aalto_username@paniikki_computer`
From personal computer
- `ssh -N -f -L localhost:8888:localhost:8889 aalto_username@kosh.aalto.fi`


