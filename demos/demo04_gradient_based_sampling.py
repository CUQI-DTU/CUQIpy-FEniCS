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
import time

#%% 1. Set up FEniCS PDE
#%% 1.1. Set up mesh
ndim = 1
nx = 20
ny = 20
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
f_exp = dl.Constant(1.0)
#f_exp = dl.Expression("4.0*exp(-pow((x[0]-c0)*s0,2))*exp(-pow((x[1]-c1)*s1,2))",c0=0.5, c1=.8, s0=10, s1=10, degree=1)
f = dl.interpolate(f_exp, solution_function_space)

im = dl.plot(f)
plt.colorbar(im)

#%% 1.5. Set up PDE variational form
def form(m,u,p):
    return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx

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

observation_operator = None#B# lambda solution : B@solution.vector().get_local()

#obsrv = observation_operator(test)

#%%
PDE = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, 
        parameter_function_space=parameter_function_space,
        solution_function_space=solution_function_space,
        dirichlet_bc=dirichlet_bc,
        adjoint_dirichlet_bc=adjoint_dirichlet_bc,
        observation_operator=observation_operator)

#%% 2.2. Create domain geometry 
fenics_continuous_geo = cuqipy_fenics.geometry.FEniCSContinuous(parameter_function_space)
domain_geometry = cuqipy_fenics.geometry.MaternExpansion(fenics_continuous_geo, length_scale = .1, num_terms=64, nu=0.1)

#%% 2.3. Create range geometry
range_geometry= cuqipy_fenics.geometry.FEniCSContinuous(solution_function_space)

#range_geometry= cuqi.geometry.Continuous1D(len(bnd_idx)) 

#%% 2.4. Create cuqi forward model
cuqi_model = cuqi.model.PDEModel(PDE, domain_geometry =domain_geometry,range_geometry= range_geometry)

#%% 2.5. Create exact solution and data and plot
#%% 2.5. Create prior
x = cuqi.distribution.Gaussian(mean=np.zeros(cuqi_model.domain_dim),
                               cov=400, 
                               geometry=domain_geometry)


#%% 2.6. Create exact solution and data and plot
exact_choice = 3
if exact_choice == 1: # Prior sample
    exact_solution =cuqi.array.CUQIarray( np.random.randn(domain_geometry.par_dim),is_par=True,geometry= domain_geometry )

elif exact_choice == 2: # Custom signal
    expr = dl.Expression("0.25*(sin(2*pi*x[0])+1.1)*(pow(r,2)<pow(x[0]-c0,2)+pow(x[1]-c1,2))+.1", r = 0.2, c0=0.6, c1=0.2, degree=1)
    exact_fun = dl.interpolate(expr, parameter_function_space)
    exact_solution =cuqi.array.CUQIarray( exact_fun,is_par=False,geometry= domain_geometry )
    im = exact_solution.plot()
    plt.colorbar(im[0])

elif exact_choice == 3: #square-circle # Custom signal
    expr = dl.Expression("9*(pow(r,2)>pow(x[0]-c0,2)+pow(x[1]-c1,2))+1.1+7*(x[0]<0.3)", r = 0.2, c0=0.8, c1=0.4, degree=1)
    exact_fun = dl.interpolate(expr, parameter_function_space)
    exact_solution =cuqi.array.CUQIarray( exact_fun,is_par=False,geometry= domain_geometry )
    im = exact_solution.plot()
    plt.colorbar(im[0])

else:
    raise ValueError('Exact choice not implemented')

data = cuqi_model(exact_solution)

#%% plot exact solution
im = exact_solution.plot()#(vmin=-2, vmax=1)
plt.colorbar(im[0])


#%% plot data
range_geometry.plot(range_geometry.par2fun(data), is_par=False)



#%% 2.7. Create likelihood
y = cuqi.distribution.Gaussian(mean=cuqi_model(x),
                               cov=np.ones(cuqi_model.range_dim)*.01**2)
y = y(y= data)


#%% 2.8. Create posterior
cuqi_posterior = cuqi.distribution.Posterior( y, x, geometry=domain_geometry)

#%% 2.9. Verify posterior gradient at a point x_i
# cuqi gradient 
x_i =cuqi.array.CUQIarray( np.random.randn(domain_geometry.par_dim),is_par=True,geometry= domain_geometry )

print("Posterior gradient (cuqi.model)")
cuqi_grad = cuqi_posterior.gradient(x_i)

# scipy gradient
print("Scipy approx")
step = 1e-11   # finite diff step
scipy_grad = optimize.approx_fprime(x_i, cuqi_posterior.logpdf, step)

# plot gradients
plt.plot(cuqi_grad)
plt.plot(scipy_grad , '--')


#%% 2.10. Sample posterior
Ns = 500
#sampler = cuqi.sampler.MALA(cuqi_posterior, scale = 2/solution_function_space.dim(),x0=np.zeros(domain_geometry.dim))
t0 = time.time()
sampler = cuqi.sampler.NUTS(cuqi_posterior)
samples = sampler.sample_adapt(Ns,Nb=100)
t1 = time.time()
print("Sampling time: ", t1-t0)

#%% 2.11. Plot samples mean
im =samples.plot_mean()
plt.colorbar(im[0])

#%% 2.12. Plot credible interval
samples.plot_ci(plot_par=True)#,exact=exact_solution)

# %% 2.13. Plot trace
samples.plot_trace()

# %%
