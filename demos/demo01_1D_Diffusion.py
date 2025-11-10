
#%% Import necessary packages
import sys
import numpy as np
import matplotlib.pyplot as plt
import cuqi
sys.path.append('../')
import cuqipy_fenics

# Fix random seed for reproducibility
np.random.seed(0)

# Define problem parameters
mapping =  'exponential'
dim = 30
N= dim + 1
L = 1
myExactSolution= 'smooth_step'

# Set up CUQIpy-FEniCS diffusion 1D test problem
observation_operator=None
SNR = 1000
model, data, problemInfo = cuqipy_fenics.testproblem.FEniCSDiffusion1D(dim=dim,
    exactSolution=myExactSolution, observation_operator=observation_operator,
    SNR=SNR, mapping=mapping, left_bc=1, right_bc=20, endpoint=L
    ).get_components()

#%% Plot data
data.plot()
plt.title('Observed data')


# %% Define prior, likelihood, and posterior
# Prior
prior = cuqi.distribution.GMRF(0.015+np.zeros(model.domain_dim), 300, 'zero', 1)

# Likelihood
sigma = np.linalg.norm(problemInfo.exactData)/SNR 
likelihood = cuqi.distribution.Gaussian(mean=model, cov=sigma**2*np.eye(model.range_dim)).to_likelihood(data)

# Posterior
posterior = cuqi.distribution.Posterior(likelihood, prior)

#%% Sample the posterior using the pCN sampler 
sampler = cuqi.sampler.PCN(posterior, initial_point=np.zeros(prior.dim), scale=0.15)
sampler.warmup(1000)
sampler.sample(5000)
samples = sampler.get_samples().burnthin(1000)

#%% Plot results
samples.plot_ci(95, plot_par = True, exact = problemInfo.exactSolution, linestyle='-', marker='.')
plt.xticks(np.arange(prior.dim)[::5],['v'+str(i) for i in range(prior.dim)][::5]);

plt.figure();
samples.plot([10,24]);
