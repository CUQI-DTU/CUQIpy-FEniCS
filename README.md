# CUQIpy-FEniCS
CUQIpy-FEniCS is a plugin for [CUQIpy](https://github.com/CUQI-DTU/CUQIpy) software package. It provides an interface between FEniCS PDE models and CUQIpy modules.

## Installation
First install [FEniCS](https://fenicsproject.org/download/archive/), we 
recommend using Anaconda from the available installation options. Then install 
CUQIpy-FEniCS with pip:
```bash
pip install cuqipy-fenics
```
If CUQIpy is not installed, it will be installed automatically. We provide 
below [additional information about FEniCS installation](#fenics_install) 
in case the above approach was problematic. 

## Quickstart
```python
import numpy as np
import matplotlib.pyplot as plt
import cuqi
import cuqipy_fenics

# Load a fenics forward model and data from testproblem library
model, y_data, info = cuqipy_fenics.testproblem.FEniCSDiffusion1D.get_components(
    dim=20,
    endpoint=1,
    exactSolution='smooth_step',
    mapping='exponential',
    SNR=10000,
    left_bc=0,
    right_bc=8
)

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

If you do not wish to install FEniCS on your machine or you have difficulty doing so, one option is to
install FEniCS on  
[Google Colaboratory (Colab) Jupyter Notebook service](https://colab.google). Below is
how to achieve that using [FEM on Colab packages](https://fem-on-colab.github.io/index.html). First open a notebook in Google Colab (an introduction to Colab can be found [here](https://colab.research.google.com/?utm_source=scs-index#scrollTo=GJBs_flRovLc)) and write in a cell:


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
