#%%
import numpy as np
import dolfin as dl
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt

#%%
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
u0_func = lambda x: 1/30*(1-np.cos(2*np.pi*(L-x)/(L)))\
                    +1/30*np.exp(-2*(10*(x-0.5))**2)+\
                     1/30*np.exp(-2*(10*(x-0.8))**2)

M = None
#library = "core"  # "core" or "fenics"
library = "fenics"  
#%%
# 2: Construct PDE and 
if library == "core":
    # Heat 1D CUQIpy Core
    grid = np.linspace(dx, L, N, endpoint=False)
    grid_obs = np.array(obs_loc)
    Dxx = (np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), -1)
        + np.diag(np.ones(N-1), 1))/dx**2 
    def PDE_form(initial_condition, t): return (Dxx, np.zeros(N),
                                                initial_condition)



    PDE = cuqi.pde.TimeDependentLinearPDE(
        PDE_form, times, grid_sol=grid, grid_obs=grid_obs)
    u0 = u0_func(grid)
    plt.plot(grid, u0, label="Initial condition")
    plt.legend()

elif library == "fenics":
    # Heat 1D CUQIpy-FEniCS
    # Heat 1D CUQIpy Core
    mesh = dl.IntervalMesh(N, x0, x1) 
    grid_obs = np.array(obs_loc)
    V = dl.FunctionSpace(mesh, "Lagrange", 1)
    u = dl.TrialFunction(V)
    v = dl.TestFunction(V)

    def PDE_form(m,u,p,t): 
        return -m*dl.inner(dl.grad(u), dl.grad(p)) * dl.dx

    Time_dependent_form = None
    initial_condition_exp = cuqipy_fenics.utilities.ExpressionFromCallable(u0_func) 

    PDE = cuqipy_fenics.pde.TimeDependentLinearFEniCSPDE(PDE_form, mesh, V,
                 V, times)


    u0 = dl.interpolate(initial_condition_exp, V)
    dl.plot(u0, title="Initial condition")


    #x_exact_raw =1/30*(1-np.cos(2*np.pi*(L-grid)/(L)))\
    #                +1/30*np.exp(-2*(10*(grid-0.5))**2)+\
    #                 1/30*np.exp(-2*(10*(grid-0.8))**2)

    #PDE = cuqi.pde.TimeDependentLinearPDE(
    #    PDE_form, times, grid_sol=grid, grid_obs=grid_obs)


else:
    raise ValueError("Unknown library: %s" % library)

#%%
# 3: Solve the PDE
if False:
    PDE.assemble(u0)
    u, _ = PDE.solve()
    u_obs = PDE.observe(u)

#%%
# 4: Plot final solution
if False:
    plt.plot(grid, u, label="Solution")
    plt.plot(grid_obs, u_obs, "o", label="Observations")
    plt.plot(grid, x_exact_raw, label="Initial condition")
    plt.legend()

# Plot the time evolution of the solution

# Plot the solution at the observation times and locations

# %%
