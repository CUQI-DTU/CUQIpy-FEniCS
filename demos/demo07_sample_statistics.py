"""
Samples Statistics Computation and Visualization
================================================
This demo illustrates computing and visualizing sample statistics
for samples that are interpreted using the cuqipy_fenics geometry.
"""

# %%
# Import the necessary modules
# ----------------------------
from cuqi.distribution import Gaussian
from cuqipy_fenics.geometry import FEniCSContinuous, MaternKLExpansion, FEniCSMappedGeometry
from cuqipy_fenics.utilities import compute_stats
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

# %%
# Define the geometries
# ---------------------
# First define the FEniCS mesh and function space.
mesh = dl.UnitSquareMesh(32, 32)
V = dl.FunctionSpace(mesh, "CG", 2) #1

# %%
# Next, define the geometries. We will define three different geometries:
# 1. A FEniCS geometry
G_FEM = FEniCSContinuous(V)

# 2. A KL expansion geometry
num_terms = 10
length_scale = 0.05
G_KL = MaternKLExpansion(G_FEM, length_scale, num_terms)

# 3. A mapped geometry applied to the KL expansion geometry
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
samples = x.sample(100)

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
samples.funvals.vector.plot_mean()
plt.title("mean of samples, G_map, function space")
plt.figure()
im = samples.funvals.vector.plot_variance(vmin=-5, vmax=25, mode="color")
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

# Interpolate variance_1 on the space V (from higher to lower dimension
# space)
print("variance_1 dim:", var1.function_space().dim())
print("variance_2 dim:", var2.function_space().dim())
print("V dim:", V.dim())
var1_interpolated = dl.interpolate(var1, V)
print("error norm: error between var1_interpolated and var2:",
      dl.errornorm(var1_interpolated, var2))
print("norm of var1_interpolated:", dl.norm(var1_interpolated))

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
samples.funvals.vector.plot_mean()
plt.title("mean of samples, G_KL, function space")
plt.figure()
im = samples.funvals.vector.plot_variance(vmin=0, vmax=0.02, mode="color")
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

# Interpolate variance_1 on the space V (from higher to lower dimension
# space)
print("variance_1 dim:", var1.function_space().dim())
print("variance_2 dim:", var2.function_space().dim())
print("V dim:", V.dim())
var1_interpolated = dl.interpolate(var1, V)
print("error norm: error between var1_interpolated and var2:",
      dl.errornorm(var1_interpolated, var2))
print("norm of var1_interpolated:", dl.norm(var1_interpolated))

# %%
# Compute sample statistics based on the geometry G_FEM
# -----------------------------------------------------

# We first generate samples using the FEM geometry.
x2 = Gaussian(0, np.ones(V.dim()), geometry=G_FEM)
samples2 = x2.sample(100)

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
samples2.funvals.vector.plot_mean()
plt.title("mean of samples, G_FEM, function space")
plt.figure()
im = samples2.funvals.vector.plot_variance(vmin=-2, vmax=8, mode="color")
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

# Interpolate variance_1 on the space V (from higher to lower dimension
# space)
print("variance_1 dim:", var1.function_space().dim())
print("variance_2 dim:", var2.function_space().dim())
print("V dim:", V.dim())
var1_interpolated = dl.interpolate(var1, V)
print("error norm: error between var1_interpolated and var2:",
      dl.errornorm(var1_interpolated, var2))
print("norm of var1_interpolated:", dl.norm(var1_interpolated))
