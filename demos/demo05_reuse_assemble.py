#%% 0. Imports required libraries and set up configuration
import dolfin as dl
import sys
import numpy as np
sys.path.append('../')
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt
import ufl
from scipy import optimize
np.random.seed(seed=1)
dl.set_log_level(40)

import cProfile



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

dirichlet__bc_expr = dl.Expression("0", degree=1) 
adjoint_dirichlet_bc_expr = dl.Constant(0.0)
dirichlet_bc = dl.DirichletBC(solution_function_space,
                              dirichlet__bc_expr,
                              u_boundary) #forward problem bcs
adjoint_dirichlet_bc = dl.DirichletBC(solution_function_space,
                                      adjoint_dirichlet_bc_expr,
                                      u_boundary) #adjoint problem bcs

#%% 1.4. Set up source term
f1 = dl.Constant(1.0)

f2 = dl.Expression("sin(2*x[0]*pi)*sin(2*x[1]*pi)", degree=1)

#%% 1.5. Set up PDE variational form
def lhs_form(m,u,p):
    return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx 

def rhs_form1(m,p):
    return - f1*p*ufl.dx

def rhs_form2(m,p):
    return - f2*p*ufl.dx

#%% 2. Set up CUQI PDE inverse probelem  

#%% 2.1. Create PDE object 
#  
boundary_elements = dl.AutoSubDomain(lambda x, on_bnd: on_bnd)
boundary_indicator = dl.DirichletBC(solution_function_space, 2, boundary_elements)


u = dl.Function(solution_function_space)
boundary_indicator.apply( u.vector() )
values = u.vector()
bnd_idx = np.argwhere( values==2 ).reshape(-1)

test = dl.Function(solution_function_space)

test_vec = np.zeros(solution_function_space.dim())
test_vec[bnd_idx]=np.ones(len(bnd_idx))
test.vector().set_local( test_vec )

dl.plot(test)

B =np.zeros((len(bnd_idx),solution_function_space.dim() ))
for idx in range(len(bnd_idx)):
    B[idx,bnd_idx[idx]] = 1 

observation_operator = None #B# lambda solution : B@solution.vector().get_local()

#obsrv = observation_operator(test)

#%%
PDE1 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE( None, mesh, 
        parameter_function_space=parameter_function_space,
        solution_function_space=solution_function_space,
        dirichlet_bc=dirichlet_bc,
        adjoint_dirichlet_bc=adjoint_dirichlet_bc,
        observation_operator=observation_operator,
        lhs_form=lhs_form,
        rhs_form=rhs_form1)

PDE2 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE( None, mesh, 
        parameter_function_space=parameter_function_space,
        solution_function_space=solution_function_space,
        dirichlet_bc=dirichlet_bc,
        adjoint_dirichlet_bc=adjoint_dirichlet_bc,
        observation_operator=observation_operator,
        lhs_form=lhs_form,
        rhs_form=rhs_form2,
        reuse_assemble_lhs=True,
        companion_model=PDE1)


#%% 2.2. Create domain geometry 
fenics_continuous_geo = cuqipy_fenics.geometry.FEniCSContinuous(parameter_function_space)
domain_geometry = cuqipy_fenics.geometry.MaternExpansion(fenics_continuous_geo, length_scale = .1, num_terms=32)

#%% 2.3. Create range geometry
range_geometry= cuqipy_fenics.geometry.FEniCSContinuous(solution_function_space)

#range_geometry= cuqi.geometry.Continuous1D(len(bnd_idx)) 

#%% 2.4. Create cuqi forward model
cuqi_model1 = cuqi.model.PDEModel(PDE1, domain_geometry =domain_geometry,range_geometry= range_geometry)

cuqi_model2 = cuqi.model.PDEModel(PDE2, domain_geometry =domain_geometry,range_geometry= range_geometry)

#%% 2.5. Create exact solution and data and plot
#%% 2.5. Create prior
x = cuqi.distribution.Gaussian(mean=np.zeros(cuqi_model1.domain_dim),
                               cov=1, geometry=domain_geometry)


#%% 2.6. Create exact solution and data and plot
exact_solution =cuqi.samples.CUQIarray( np.random.randn(domain_geometry.par_dim),is_par=True,geometry= domain_geometry )

data1 = cuqi_model1(exact_solution)
data2 = cuqi_model2(exact_solution)

#%% plot exact solution
im = exact_solution.plot(vmin=-2, vmax=1)
plt.colorbar(im[0])


#%% plot data 1
range_geometry.plot(range_geometry.par2fun(data1), is_par=False)

#%% plot data 2
range_geometry.plot(range_geometry.par2fun(data2), is_par=False)


#%% 2.7. Create likelihood 1
y1 = cuqi.distribution.Gaussian(mean=cuqi_model1(x),
                               cov=np.ones(cuqi_model1.range_dim)*.01**2)
y1 = y1(y1= data1)

#%% 2.8. Create likelihood 2
y2 = cuqi.distribution.Gaussian(mean=cuqi_model2(x),
                                 cov=np.ones(cuqi_model2.range_dim)*.01**2) 
y2 = y2(y2= data2)

#%% 2.9. Create posterior
cuqi_posterior = cuqi.distribution.JointDistribution( y1, y2, x)._as_stacked()
#cuqi_posterior.geometry = domain_geometry



#%% 2.10. Sample posterior
Ns = 100
#sampler = cuqi.sampler.MALA(cuqi_posterior, scale = 2/solution_function_space.dim(),x0=np.zeros(domain_geometry.dim))
sampler = cuqi.sampler.MetropolisHastings(cuqi_posterior)


cProfile.run('samples = sampler.sample_adapt(Ns,Nb=10)', filename='profile.out')
samples.geometry = domain_geometry

#%% 2.11 Sample without reusing lhs
PDE2.reuse_assemble_lhs = False
PDE2.companion_model = None
cProfile.run('samples = sampler.sample_adapt(Ns,Nb=10)', filename='profile_no_reuse.out')
samples.geometry = domain_geometry

#%% 2.11. Plot samples mean
im =samples.plot_mean()
plt.colorbar(im[0])

#%% 2.12. Plot credible interval
samples.plot_ci(plot_par=True,exact=exact_solution)

# %% 2.13. Plot trace
samples.plot_trace()
