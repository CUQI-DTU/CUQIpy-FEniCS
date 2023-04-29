"""
Samples Statistics Computation and Visualization
================================================
This demo illustrates computing and visualizing sample statistics
for samples that are interpreted using the cuqipy-fenics geometry.
"""

# %%
# Import the necessary modules
# ----------------------------
from cuqi.distribution import Gaussian
from cuqipy_fenics.geometry import FEniCSContinuous, MaternExpansion, FEniCSMappedGeometry
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
import warnings

# %% Define helper function 
# -------------------------
def compute_stats(samples):
    """This function computes the statistics (mean and variance) of a set of 
    samples on the function value representation. Two different approaches 
    used in computing the variance: The first approach is computing the variance
    var1 using the FEniCS functions directly, and is general for any FEM 
    function. The second approach is computing the variance var2 using the
    FEniCS vectors directly, and is specific for some FEM function spaces,
    e.g. CG1. The returned values are the mean, var1, and var2."""

    geom = samples.geometry
    V = geom.function_space

    # Loop to compute the samples function value
    sample_funs = []
    sample_funs_dof_vecs = np.empty((V.dim(), samples.samples.shape[1]))   
    for i, sample in enumerate(samples):
        sample_funs.append(geom.par2fun(sample))
        sample_funs_dof_vecs[:, i] = sample_funs[-1].vector().get_local()

    # Sample mean
    sample_mean_dof = np.mean(sample_funs_dof_vecs, axis=1)
    sample_mean_f = dl.Function(V)
    sample_mean_f.vector().set_local(sample_mean_dof)

    # Compute variance 
    # Approach 1 (this approach is general for all FEM function spaces)
    # Loop to create terms required for variance computation 
    var_terms = np.empty((V.dim(), samples.samples.shape[1]))   
    for i, sample_f in enumerate(sample_funs):
        expr_f = dl.project(sample_f*sample_f - 2*sample_mean_f*sample_f, V)
        var_terms[:, i] = expr_f.vector().get_local()
    
    mean_var_terms = np.mean(var_terms, axis=1)
    mean_var_terms_f = dl.Function(V)
    mean_var_terms_f.vector().set_local(mean_var_terms)
    
    var1 = dl.project(mean_var_terms_f + sample_mean_f*sample_mean_f , V)

    # Approach 2 (this approach is specific for some FEM function spaces, e.g. CG1)
    if V.ufl_element().family() != 'Lagrange':
        warnings.warn("The function space is not Lagrange, the variance, var2,"+ 
                     "computed using the second approach may not be correct.")
    var2_vec = np.var(sample_funs_dof_vecs, axis=1)
    var2 = dl.Function(V)
    var2.vector().set_local(var2_vec)

    return sample_mean_f, var1, var2


# %%
# Define the geometries
# -------------------
# First define the FEniCS mesh and function space.
mesh = dl.UnitSquareMesh(64, 64)
V = dl.FunctionSpace(mesh, "CG", 2) #1

# %%
# Next, define the geometries. We will define three different geometries:
# 1. A FEniCS geometry
G_FEM = FEniCSContinuous(V)

# 2. A KL expansion geometry
num_terms = 10
length_scale = 0.05
G_KL = MaternExpansion(G_FEM, length_scale, num_terms)

# 3. A mapping geometry applied to the KL expansion geometry
c_minus = 1
c_plus = 10
def heavy_map(func):
    dofs = func.vector().get_local()
    updated_dofs = c_minus*0.5*(1 + np.sign(dofs)) + c_plus*0.5*(1 - np.sign(dofs))
    func.vector().set_local(updated_dofs)
    return func
G_map = FEniCSMappedGeometry(G_KL, heavy_map)

# %%
# Create a distribution and sample the distribution
# -------------------------------------------------
# We then create a random field of mapped field represented by the KL expansion.
x = Gaussian(0, np.ones(num_terms), geometry=G_map)

# %%
# We use this random field to generate samples.
samples = x.sample(3)

# %% 
# Compute sample statistics based on the geometry G_map
# -----------------------------------------------------
# mean and variance computed on the **parameter** space then mapped to the 
# function space
plt.figure()
samples.plot_mean()
plt.title("mean of samples, G_map, parameter space")
plt.figure()
samples.plot_variance()
plt.title("variance of samples, G_map, parameter space")

# mean and variance computed on the **function** space directly
plt.figure()
samples.funvals.plot_mean()
plt.title("mean of samples, G_map, function space")
plt.figure()
im = samples.funvals.plot_variance(vmin=-5, vmax=25, mode="color")
plt.title("variance of samples, G_map, function space")
plt.colorbar(im[0])

# variance computed on the **function** space using the helper function
mean_f, var1, var2 = compute_stats(samples)
plt.figure()
dl.plot(mean_f, title="mean of samples, G_map, function space, computed using helper function")
plt.figure()
im = dl.plot(var1, title="variance_1 of samples, G_map, function space, computed using helper function", vmin=-5, vmax=25, mode="color")
plt.colorbar(im)
plt.figure()
im = dl.plot(var2, title="variance_2 of samples, G_map, function space, computed using helper function", vmin=-5, vmax=25, mode="color")
plt.colorbar(im)


# %% 
# Compute sample statistics based on the geometry G_KL
# -----------------------------------------------------
# mean and variance computed on the **parameter** space then mapped to the
# function space
plt.figure()
samples.geometry = G_KL
samples.plot_mean()
plt.title("mean of samples, G_KL, parameter space")
plt.figure()
samples.plot_variance()
plt.title("variance of samples, G_KL, parameter space")

# mean and variance computed on the **function** space directly
plt.figure()
samples.funvals.plot_mean()
plt.title("mean of samples, G_KL, function space")
plt.figure()
im = samples.funvals.plot_variance(vmin=0, vmax=0.02, mode="color")
plt.title("variance of samples, G_KL, function space")
plt.colorbar(im[0])

# variance computed on the **function** space using the helper function
mean_f, var1, var2 = compute_stats(samples)
plt.figure()
dl.plot(mean_f, title="mean of samples, G_KL, function space, computed using helper function")
plt.figure()
im = dl.plot(var1, title="variance_1 of samples, G_KL, function space, computed using helper function", vmin=0, vmax=0.02, mode="color")
plt.colorbar(im)
plt.figure()
im = dl.plot(var2, title="variance_2 of samples, G_KL, function space, computed using helper function", vmin=0, vmax=0.02, mode="color")
plt.colorbar(im)

# %%
# Compute sample statistics based on the geometry G_FEM
# -----------------------------------------------------

# We first generate samples using the FEM geometry.
x2 = Gaussian(0, np.ones(V.dim()), geometry=G_FEM)
samples2 = x2.sample(3)

# mean and variance computed on the **parameter** space then mapped to the
# function space
plt.figure()
samples2.geometry = G_FEM
samples2.plot_mean()
plt.title("mean of samples, G_FEM, parameter space")
plt.figure()
samples2.plot_variance()
plt.title("variance of samples, G_FEM, parameter space")

# mean and variance computed on the **function** space directly
plt.figure()
samples2.funvals.plot_mean()
plt.title("mean of samples, G_FEM, function space")
plt.figure()
im = samples2.funvals.plot_variance(vmin=-2, vmax=8, mode="color")
plt.title("variance of samples, G_FEM, function space")
plt.colorbar(im[0])

# variance computed on the **function** space using the helper function
mean_f, var1, var2 = compute_stats(samples2)
plt.figure()
dl.plot(mean_f, title="mean of samples, G_FEM, function space, computed using helper function")
plt.figure()
im = dl.plot(var1, title="variance_1 of samples, G_FEM, function space, computed using helper function", vmin=-2, vmax=8, mode="color")
plt.colorbar(im)
plt.figure()
im = dl.plot(var2, title="variance_2 of samples, G_FEM, function space, computed using helper function", vmin=-2, vmax=8, mode="color")
plt.colorbar(im)
