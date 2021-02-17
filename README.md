[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/wells/HEAD?filepath=well_test.ipynb)

# wells
Jupyter Notebooks for teaching well testing

## Cloud Access
Use the [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/wells/HEAD?filepath=well_test.ipynb) badge to open a binder notebook.

## Local Installation

Ensure you have Anaconda Python 3.7 installed. Then

1. Clone the repo

```bash
git clone https://github.com/ddempsey/wells
```

2. CD into the repo and create a conda environment

```bash
cd wells

conda env create -f environment.yml

conda activate wells
```

3. Add the conda environment so it is accessible from the Jupyter Notebook

```bash
python -m ipykernel install --user --name=wells
```

## Use

Open a Jupyter Notebook server from the terminal

```bash
jupyter notebook
```

Open the `well_test.ipynb` notebook and select the `wells` environment in `Kernel > Change kernel`.

Run the notebook cells.
