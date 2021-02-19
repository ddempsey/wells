[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/wells/HEAD?filepath=well_test.ipynb)

# Wells
Jupyter Notebooks for teaching well testing.

## Cloud Access
Use the [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/wells/HEAD?filepath=well_test.ipynb) badge to open a binder notebook.

## Local Installation

Ensure you have Anaconda Python 3.X installed. Then

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

If you are a local user, open a Jupyter Notebook server from the terminal

```bash
jupyter notebook
```

In the local server, or via the binder linke, open the `well_test.ipynb` notebook. In the local server, select the `wells` environment in `Kernel > Change kernel`.

Run the notebook cells.

A document has been included in the repository with questions to test your understanding of the pumping test concepts.

## Author

David Dempsey (Civil and Natural Resource Engineering, University of Canterbury)
