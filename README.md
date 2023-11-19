# CUQIpy-FEniCS
CUQIpy-FEniCS is a plugin for [CUQIpy](https://github.com/CUQI-DTU/CUQIpy) software package. It provides an interface between FEniCS PDE models and CUQIpy modules. The documentation for CUQIpy-FEniCS can be found [here](https://cuqi-dtu.github.io/CUQIpy-FEniCS).

## Installation
First install [FEniCS](https://fenicsproject.org/download/archive/), we 
recommend using Anaconda from the available installation options. Then install 
CUQIpy-FEniCS with pip:
```bash
pip install cuqipy-fenics
```
If CUQIpy is not installed, it will be installed automatically. We provide 
[additional information about FEniCS installation](#fenics_install), below,
in case the above approach was problematic. 

## Quickstart
```python
import numpy as np
import matplotlib.pyplot as plt
import cuqi
import cuqipy_fenics

# Load a fenics forward model and data from testproblem library
model, y_data, info = cuqipy_fenics.testproblem.FEniCSDiffusion1D(
    dim=20,
    endpoint=1,
    exactSolution='smooth_step',
    mapping='exponential',
    SNR=10000,
    left_bc=0,
    right_bc=8
).get_components()

# Set up Bayesian model
x = cuqi.distribution.GMRF(np.zeros(model.domain_dim),
                           25, 1, 'zero', geometry=model.domain_geometry)
# y ~ N(model(x), 0.01^2)
y = cuqi.distribution.Gaussian(mean=model(x), cov=0.05**2)

# Set up Bayesian Problem object
BP = cuqi.problem.BayesianProblem(y, x).set_data(y=y_data)

# Sample from the posterior
samples = BP.sample_posterior(5000)

# Analyze the samples
samples.burnthin(1000).plot_ci(95, plot_par=True,
                               exact=info.exactSolution, linestyle='-', marker='.')
```

For more examples, see the [demos](demos) folder.

<a id="fenics_install"></a>
## More on using and installing FEniCS


### FEniCS on Google Colaboratory

If you do not wish to install FEniCS on your machine or you have difficulty 
doing so, one option is to install FEniCS on [Google Colaboratory (Colab) Jupyter Notebook service](https://colab.google). 
We show below how to achieve this using 
[FEM on Colab packages](https://fem-on-colab.github.io/index.html). 
First open a notebook in Google Colab (an introduction to Colab can be found
[here](https://colab.research.google.com/?utm_source=scs-index#scrollTo=GJBs_flRovLc))
and write in a cell:


```
try:
    import dolfin
except ImportError:
    !wget "https://fem-on-colab.github.io/releases/fenics-install.sh" -O "/tmp/fenics-install.sh" && bash "/tmp/fenics-install.sh"
    import dolfin
```
Then in a following cell use `pip` to install `cuqipy-fenics` as follows:

```
!pip install cuqipy-fenics
```
Test that you can import `cuqi` and  `cuqipy_fenics` 

```
import cuqi
import cuqipy_fenics
```

### Create a Docker image for FEniCS 
Here is another approach of installing FEniCS that could be useful in
some cases (e.g. MAC M1 machines). We create a Docker image that contains a 
conda environment for FEniCS.

Run the following command lines to create a directory named, for example,
`my_dir`.


```
mkdir my_dir
cd my_dir
```

In `my_dir`, run the following command lines to create the files needed for the 
Docker image and to pull a base image `continuumio/miniconda3`.

```
touch Dockerfile
touch environment.yml
docker pull continuumio/miniconda3
```

Edit the file `Dockerfile` you just created to have the following lines: 
```
# Define base image
FROM continuumio/miniconda3
 
# Set working directory for the project
WORKDIR /app
 
# Create Conda environment from the YAML file
COPY environment.yml .
RUN conda env create -f environment.yml
```

Edit the file `environment.yml` you just created to have the following lines
```
name: env
channels:
   - conda-forge
dependencies:
   - python=3.10
   - fenics
   - pip
   - pip:
   # works for regular pip packages
     - cuqipy-fenics
```
(consider also `python=3.8` if `python=3.10` turn out to be problematic)

Build the docker image, named `my_image` for example, by running the following command inside `my_dir` (this might take a little while):
```
docker build -t my_image .
```

Run the docker image:
```
docker run --entrypoint=/bin/bash -it -p 127.0.0.1:8090:8000 -v $(pwd):/app -w /app my_image
```

Inside the Docker container, activate the conda environment and run python:
```
conda activate env
python
```

Run the following python commands to make sure `FEniCS` and `cuqipy-fenics` are
installed
```
import fenics
import cuqi
import cuqipy_fenics
```
