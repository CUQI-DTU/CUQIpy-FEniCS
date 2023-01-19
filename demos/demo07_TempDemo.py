#%%
import numpy as np
import dolfin as dl
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt

# 1: Common settings
x0 = 0
x1 = 1
L = x1 - x0
N = 100
dx = L / N
t0 = 0
t1 = 0.01
times = np.linspace(t0, t1, 450)
obs_times = np.linspace(t0, t1, 10)
obs_loc = [0.2, 0.7]
g1 = lambda t: np.exp(-t)
g2 = 0
u0 = lambda x: np.exp(-10 * (x - 0.5) ** 2)

M = None
library = "core" # "core" or "fenics"

# Construct PDE and 
if library == "core":
    # Heat 1D CUQIpy Core
    grid = np.linspace(dx, L, N, endpoint=False)
    grid_obs = np.array(obs_loc)
    Dxx = (np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), -1)
        + np.diag(np.ones(N-1), 1))/dx**2 
    def PDE_form(initial_condition, t): return (Dxx, np.zeros(N),
                                                initial_condition)

    x_exact_raw =1/30*(1-np.cos(2*np.pi*(L-grid)/(L)))\
                    +1/30*np.exp(-2*(10*(grid-0.5))**2)+\
                     1/30*np.exp(-2*(10*(grid-0.8))**2)

    PDE = cuqi.pde.TimeDependentLinearPDE(
        PDE_form, times, grid_sol=grid, grid_obs=grid_obs)

    PDE.assemble(x_exact_raw)
    u, _ = PDE.solve()
    u_obs = PDE.observe(u)

elif library == "fenics":
    # Heat 1D CUQIpy-FEniCS
    pass

else:
    raise ValueError("Unknown library: %s" % library)

#%%
# Plot final solution
plt.plot(grid, u, label="Solution")
plt.plot(grid_obs, u_obs, "o", label="Observations")
plt.plot(grid, x_exact_raw, label="Initial condition")
plt.legend()

# Plot the time evolution of the solution

# Plot the solution at the observation times and locations

# %%
