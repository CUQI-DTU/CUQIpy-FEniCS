#%% 0. Imports required libraries and set up configuration
import dolfin as dl
import sys
import numpy as np
sys.path.append('../')
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt
import ufl
dl.set_log_level(40)
import time


#%% 1. Set up FEniCS PDE
#%% 1.1. Set up mesh
ndim = 1
nx = 40
ny = 40
mesh = dl.UnitSquareMesh(nx, ny)

#%% 1.2. Set up function spaces
solution_function_space = dl.FunctionSpace(mesh, 'Lagrange', 2)
parameter_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1)

#%% 1.3. Set up Dirichlet boundaries
def u_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

dirichlet_bc_expr = dl.Expression("0", degree=1) 
adjoint_dirichlet_bc_expr = dl.Constant(0.0)
dirichlet_bc = dl.DirichletBC(solution_function_space,
                              dirichlet_bc_expr,
                              u_boundary) #forward problem bcs

#%% 1.4. Set up two different source terms
f1 = dl.Constant(1.0)
f2 = dl.Expression("sin(2*x[0]*pi)*sin(2*x[1]*pi)", degree=1)

#%% 1.5. Set up PDE variational forms
def lhs_form(m,u,p):
    return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx 

def rhs_form1(m,p):
    return - f1*p*ufl.dx

def rhs_form2(m,p):
    return - f2*p*ufl.dx

#%% 2. Set up CUQI PDE Bayesian problem  
#%% 2.1. Create two PDE objects with different source terms 
PDE1 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE( (lhs_form, rhs_form1), mesh, 
        parameter_function_space=parameter_function_space,
        solution_function_space=solution_function_space,
        dirichlet_bc=dirichlet_bc,
        observation_operator=None,
        reuse_assembled=False)

PDE2 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE( (lhs_form, rhs_form2), mesh, 
        parameter_function_space=parameter_function_space,
        solution_function_space=solution_function_space,
        dirichlet_bc=dirichlet_bc,
        observation_operator=None,
        reuse_assembled=False)

#%% 2.2. Create domain geometry 
fenics_continuous_geo = cuqipy_fenics.geometry.FEniCSContinuous(
    parameter_function_space)
domain_geometry = cuqipy_fenics.geometry.MaternKLExpansion(
    fenics_continuous_geo, length_scale = .1, num_terms=32)

#%% 2.3. Create range geometry
range_geometry= cuqipy_fenics.geometry.FEniCSContinuous(solution_function_space)

#%% 2.4. Create cuqi forward model (two models corresponding to two PDE objects)
cuqi_model1 = cuqi.model.PDEModel(
    PDE1, domain_geometry =domain_geometry,range_geometry=range_geometry)

cuqi_model2 = cuqi.model.PDEModel(
    PDE2, domain_geometry =domain_geometry,range_geometry=range_geometry)

#%% 2.5. Create the prior
x = cuqi.distribution.Gaussian(mean=np.zeros(cuqi_model1.domain_dim),
                               cov=1, geometry=domain_geometry)


#%% 2.6. Create exact solution and data
exact_solution =cuqi.array.CUQIarray( 
    np.random.randn(domain_geometry.par_dim),
    is_par=True,geometry= domain_geometry )

data1 = cuqi_model1(exact_solution)
data2 = cuqi_model2(exact_solution)

#%% 2.7. plot
#%% plot the exact solution
im = exact_solution.plot()
plt.colorbar(im[0])

#%% plot data 1
range_geometry.plot(data1, is_par=True)

#%% plot data 2
range_geometry.plot(data2, is_par=True)

#%% 2.8. Create likelihood 1
y1 = cuqi.distribution.Gaussian(mean=cuqi_model1(x),
                               cov=np.ones(cuqi_model1.range_dim)*.01**2)
y1 = y1(y1= data1)

#%% 2.9. Create likelihood 2
y2 = cuqi.distribution.Gaussian(mean=cuqi_model2(x),
                                 cov=np.ones(cuqi_model2.range_dim)*.01**2) 
y2 = y2(y2= data2)

#%% 2.10. Create posterior
cuqi_posterior = cuqi.distribution.JointDistribution( y1, y2, x)._as_stacked()


#%% 3 Solve the Bayesian problem
#%% 3.1. Sample the posterior (Case 1: no reuse of assembled operators)
Ns = 100
np.random.seed(0) # fix seed for reproducibility 
sampler = cuqi.sampler.MetropolisHastings(cuqi_posterior)
t0 = time.time()
samples1 = sampler.sample_adapt(Ns,Nb=10)
t1 = time.time()
print('Time elapsed: (Case 1: no reuse of assembled operators)', t1-t0, 's')
samples1.geometry = domain_geometry

#%% 3.2. Set PDE2 to be a shallow copy of PDE1 but with different rhs
PDE1.reuse_assembled = True
PDE2 = PDE1.with_updated_rhs(rhs_form2)
cuqi_model2.pde = PDE2

#%% 3.3. Sample the posterior again (Case 2: reuse of assembled operators)
np.random.seed(0)
sampler = cuqi.sampler.MetropolisHastings(cuqi_posterior)
t0 = time.time()
samples2 = sampler.sample_adapt(Ns,Nb=10)
t1 = time.time()
print('Time elapsed (Case 2: reuse of assembled operators): ', t1-t0, 's')
samples2.geometry = domain_geometry

#%% 3.4. Plot samples mean
im = samples1.plot_mean()
plt.colorbar(im[0])

#%% 3.5. Plot credible interval
samples1.plot_ci(plot_par=True,exact=exact_solution)

# %% 3.6. Plot trace
samples1.plot_trace()

# %% 3.7. Assert that the samples are the same Case 1 and Case 2
print(np.allclose(samples1.samples, samples2.samples))
