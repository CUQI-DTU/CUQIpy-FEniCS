import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
import cuqipy_fenics
import dolfin as dl
import warnings
import cuqi
from cuqi.problem import BayesianProblem
from cuqi.model import PDEModel
from cuqi.distribution import Gaussian
from cuqi.geometry import Geometry
import dolfin as dl
import ufl
from .pde import SteadyStateLinearFEniCSPDE
from .geometry import FEniCSContinuous, FEniCSMappedGeometry,\
      MaternExpansion
from .utilities import to_dolfin_expression


__all__ = ['FEniCSDiffusion1D', 'FEniCSPoisson2D']

class FEniCSDiffusion1D(BayesianProblem):
    """
    1D Diffusion PDE problem using FEniCS. The problem has Dirichlet boundary conditions.

    Parameters
    ------------
    dim : int, Default 100
        Dimension of the 1D problem

    endpoint : float, Default 1
        Endpoint of the 1D grid (starts at 0).

    exactSolution : ndarray, CUQIarray, Default None
        Exact solution used to generate data. 
        If None a default exact solution is chosen.

    SNR : float, Default 100
        Signal-to-noise ratio.
    
    mapping: str or callable
        mapping to parametrize the Bayesian parameters. If provided as string, it can take one of the values: ['exponential']

    Attributes
    ----------
    data : ndarray
        Generated (noisy) data

    model : cuqi.model.Model
        The forward model

    prior : cuqi.distribution.Distribution
        Distribution of the prior (Default = None)

    likelihood : cuqi.likelihood.Likelihood
        Likelihood function.

    exactSolution : ndarray
        Exact solution (ground truth)

    exactData : ndarray
        Noise free data

    infoSring : str
        String with information about the problem, noise etc.

    Methods
    ----------
    MAP()
        Compute MAP estimate of posterior.
        NB: Requires prior to be defined.

    sample_posterior(Ns)
        Sample Ns samples of the posterior.
        NB: Requires prior to be defined.

    """
    
    def __init__(self, dim = 100, endpoint = 1, exactSolution = None, SNR = 100, observation_operator = None, mapping = None, left_bc=0, right_bc=1):

        # Create FEniCSPDE        
        def u_boundary(x, on_boundary):
            return on_boundary

        def form(m,u,p):
            return m*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx
        
        mesh = dl.IntervalMesh(dim, 0,endpoint)

        solution_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1)
        parameter_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1)

        dirichlet_bc_expression = dl.Expression("left_bc*(x[0]<eps)+right_bc*(x[0]>endpoint-eps)", eps=dl.DOLFIN_EPS, endpoint=endpoint, left_bc=left_bc, right_bc=right_bc, degree=1)
        dirichlet_bc = dl.DirichletBC(solution_function_space, dirichlet_bc_expression, u_boundary)
        adjoint_dirichlet_bc = dl.DirichletBC(
            solution_function_space, dl.Constant(0), u_boundary)

        PDE = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
            form, mesh, solution_function_space, parameter_function_space, dirichlet_bc, adjoint_dirichlet_bc, observation_operator=observation_operator)
        
        # Create PDE model
        domain_geometry = cuqipy_fenics.geometry.FEniCSContinuous(parameter_function_space)
        if mapping is not None:
            if mapping == 'exponential':
                mapping = lambda x : ufl.exp(x)
            elif callable(mapping):
                mapping = mapping
            else:
                raise ValueError('mapping can be a callable or one of the expected keywords.')
            domain_geometry =  cuqipy_fenics.geometry.FEniCSMappedGeometry(geometry=domain_geometry, map = mapping)

        range_geometry = cuqipy_fenics.geometry.FEniCSContinuous(solution_function_space)
        
        model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)

        # Create prior
        pr_mean = np.zeros(domain_geometry.par_dim)
        x = cuqi.distribution.GMRF(pr_mean,25,1,'zero') 
        
        # Set up exact solution
        if exactSolution is None:
            exactSolution = x.sample()
        elif exactSolution == 'smooth_step':
            N = dim + 1
            fun = lambda grid:  0.8*np.exp( -( (grid -endpoint/2.0 )**2 ) / 0.02)
            grid = np.linspace(0,endpoint,N)
            exactSolution = np.ones(N)*.8
            exactSolution[np.where(grid > endpoint/2.0)
                          ] = fun(grid[np.where(grid > endpoint/2.0)])
            exactSolution = cuqi.array.CUQIarray(
                exactSolution, geometry=domain_geometry)

        # Generate exact data
        b_exact = model.forward(domain_geometry.par2fun(exactSolution),is_par=False)

        # Add noise to data
        # Reference: Adding noise with a desired signal-to-noise ratio
        # https://sites.ualberta.ca/~msacchi/SNR_Def.pdf
        noise = np.random.normal(0, 1, b_exact.shape)
        alpha = np.linalg.norm(b_exact)/(np.sqrt(SNR)*np.linalg.norm(noise))

        data = cuqi.array.CUQIarray(
            b_exact + alpha*noise, geometry=range_geometry)

        # Create likelihood
        y = cuqi.distribution.Gaussian(
            mean=model(x), cov=alpha*np.eye(range_geometry.par_dim))

        # Initialize FEniCSDiffusion1D as BayesianProblem problem
        super().__init__(y, x, y=data)

        # Store exact values and information
        self.exactSolution = exactSolution
        self.exactData = b_exact
        self.infoString = f"Noise type: Additive i.i.d. noise with mean zero and signal to noise ratio: {SNR}"


class FEniCSPoisson2D(BayesianProblem):
    """
    2D Diffusion PDE-based Bayesian inverse problem that uses FEniCS. 
    The problem is sat up on a unit square mesh with either Dirichlet
    or Neumann boundary conditions on each boundary. The unknown parameter
    is the (possibly heterogeneous) diffusion coefficient (e.g. conductivity)
    field. The unknown parameter (e.g. conductivity) and the PDE solution 
    (e.g. the potential) are approximated in the first order Lagrange FEM space.

    Parameters
    -----------
    dim : tuple, Default (32,32)
        | Number of the 2D mesh vertices on the x and y directions, respectively.

    bc_type : list of str, Default ['Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet']
        | Boundary conditions on each boundary. The accepted values are:
        | 'Dirichlet': Dirichlet boundary condition.
        | 'Neumann': Neumann boundary condition.
        | The list should be ordered as follows: [left, bottom, right, top]

    bc_value : list of entries, each is a float or a callable , Default [0, 0, 0, 0]
        | Boundary condition values on each boundary. The accepted values are:
        | A float: a constant value.
        | A callable : a callable that takes coordinate value list as input and return the boundary condition value at the corresponding point, e.g. `lambda x: np.sin(x[0])+np.cos(x[1])`.
        | The list should be ordered as follows: [left, bottom, right, top]

    exactSolution : ndarray, CUQIarray, or callable , Default None
        | Exact solution to the Bayesian inverse problem used to generate data, the diffusivity coefficient field in this case. When passed as a callable, it should take coordinate value list as input and return the exact solution at the corresponding point, `lambda x: np.sin(x[0])+np.cos(x[1])`. If None, a default exact solution is chosen.

    f : float, callable, or dolfin.Expression, Default 1
        | Source term in the PDE. The accepted values are:
        | A float: a constant value.
        | A callable: a callable that takes coordinate value list as input and returns the source term at the corresponding point `lambda x: np.sin(x[0])+np.cos(x[1])`.
        | A dolfin.Expression: a dolfin.Expression object that defines the source term.
        
    relative_noise_std : float, default 0.01
        Standard deviation of the noise relative to the exact data. By default, the noise is 1% (=0.01) of the exact data.

    field_type : str, Default None
        | Field type of the forward model domain. The accepted values are:
        | "KL": a :class:`MaternExpansion` geometry object will be created and set as a domain geometry.
        | None: a :class:`FEniCSContinuous` geometry object will be created and set as a domain geometry.

    field_params : dict, Default None
        | A dictionary of keyword arguments that the underlying geometry accepts. (Passed to the underlying geometry when field type is "KL" or None). For example, for "KL" field type, the dictionary can be `{"length_scale": 0.1, "num_terms": 32}`. If None is passed as field_type, this argument is ignored.

    mapping : str or callable , Default None
        | mapping to parametrize the Bayesian parameters. If None, no mapping is applied. If provided as callable, it should take a FEniCS function (of the unknown parameter) as input and return a FEniCS form, e.g. `lambda m: ufl.exp(m)`.
        If provided as string, it can take one of the values: 
        | 'exponential' : Parameterization in which the unknown parameter becomes the log of the diffusion coefficient.

    prior : cuqi.distribution.Distribution, Default Gaussian
        | Distribution of the prior. Needs to be i.i.d standard Gaussian if field_type is "KL". The prior name property, i.e., `prior.name` is expected to be "x".
    """

    def __init__(self, dim=None, bc_types=None, bc_values=None,
                 exactSolution=None, f=None, relative_noise_std=None, field_type=None,
                 field_params=None, mapping=None, prior=None):

        # Create the mesh
        if dim is None:
            dim = (32, 32)
        mesh = dl.UnitSquareMesh(dim[0], dim[1])

        # Create the function space
        V = dl.FunctionSpace(mesh, 'Lagrange', 1)

        # Set up boundary conditions
        if bc_types is None:
            bc_types = ['Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet']
        elif len(bc_types) != 4:
            raise ValueError(
                "The length of bc_types list should be 4. The list should be ordered as follows: [left, bottom, right, top]")
        elif all(bc_type.lower() in ['neumann'] for bc_type in bc_types):
            raise ValueError(
                "All boundary conditions cannot be Neumann. At least one boundary condition should be Dirichlet.")
        if bc_values is None:
            bc_values = [0, 0, 0, 0]
        elif len(bc_values) != 4:
            raise ValueError(
                "The length of bc_values list should be 4. The list should be ordered as follows: [left, bottom, right, top]")
        bc_values = [to_dolfin_expression(bc_value) for bc_value in bc_values]

        subdomains = self._create_boundaries_subdomains()
        dirichlet_bcs = self._set_up_dirichlet_bcs(
            V, bc_types, bc_values, subdomains)
        neumann_bcs = self._set_up_neumann_bcs(
            V, bc_types, bc_values, subdomains)

        # Set up the source term
        if f is None:
            f = dl.Constant(1)
        else:
            f = to_dolfin_expression(f)

        # Set up the variational problem form
        if mapping is None:
            def parameter_form(m): return m
        elif callable(mapping):
            parameter_form = mapping
        elif mapping.lower() == 'exponential':
            def parameter_form(m): return ufl.exp(m)
        else:
            raise ValueError('mapping should be a callable, None or a string.')

        def form(m, u, p):
            return parameter_form(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx\
                - f*p*ufl.dx\
                - neumann_bcs(m, p)

        # Create the CUQI PDE object
        PDE = SteadyStateLinearFEniCSPDE(
            form,
            mesh,
            parameter_function_space=V,
            solution_function_space=V,
            dirichlet_bcs=dirichlet_bcs)

        # Create the domain geometry
        G_FEM = FEniCSContinuous(V)
        if field_params is None:
            field_params = {}
        if field_type is None:
            G_domain = G_FEM
        elif field_type == 'KL':
            if field_params == {}:
                field_params = {'length_scale': 0.1, 'num_terms': 32}
            G_domain = MaternExpansion(G_FEM, **field_params)
        else:
            raise ValueError('Unknown field type.')

        # Create the range geometry
        G_range = FEniCSContinuous(V)

        # Create the forward model
        A = PDEModel(PDE, domain_geometry=G_domain, range_geometry=G_range)

        # Create the prior
        if prior is None:
            prior = Gaussian(np.zeros(A.domain_dim), 1,
                             geometry=G_domain, name='x')
        elif prior.name != 'x':
            raise ValueError('Prior name is expected to be "x".')

        # Set up the exact solution
        if exactSolution is None and field_type == "KL":
            np.random.seed(15)
            exactSolution = np.random.randn(G_domain.par_dim)
        elif exactSolution is None:
            def exactSolution(x): return 1.5 + 0.5 * \
                np.sin(2*np.pi*x[0])*np.sin(2*np.pi*x[1])

        if isinstance(exactSolution, np.ndarray):
            exactSolution = cuqi.array.CUQIarray(
                exactSolution,
                is_par=True,
                geometry=G_domain)

        elif callable(exactSolution):
            exactSolution_expr = to_dolfin_expression(exactSolution)
            exactSolution_func = dl.interpolate(exactSolution_expr, V)
            exactSolution = cuqi.array.CUQIarray(
                exactSolution_func,
                is_par=False,
                geometry=G_domain)
            
        else:
            raise ValueError('exactSolution should be a numpy array, a function or None.')
        print(exactSolution.__class__)
        
        # Create the exact data
        exact_data = A(exactSolution)
        if not isinstance(exact_data, cuqi.array.CUQIarray):
            exact_data = cuqi.array.CUQIarray(exact_data, is_par=True, geometry=G_range)

        # Create the data distribution and the noisy data
        noise = np.random.randn(len(exact_data))
        if relative_noise_std is None:
            relative_noise_std = 0.01
        noise_std = relative_noise_std * \
            np.linalg.norm(exact_data)/np.linalg.norm(noise)
        noise = noise_std*noise
        data = exact_data + noise

        y = Gaussian(mean=A(prior), cov=noise_std**2, geometry=G_range)

        # Create the Bayesian problem
        super().__init__(y, prior, y=data)

        # Store exact values and information
        self.exactSolution = exactSolution
        self.exactData = exact_data
        self.infoString = f"Noise type: Additive i.i.d. noise with standard deviation: {noise_std} and relative noise standard deviation: {relative_noise_std}."

    def _create_boundaries_subdomains(self):
        """
        Create subdomains for the boundary conditions.
        """
        class Left(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < dl.DOLFIN_EPS

        class Bottom(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] < dl.DOLFIN_EPS

        class Right(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] > 1.0 - dl.DOLFIN_EPS

        class Top(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] > 1.0 - dl.DOLFIN_EPS

        return [Left(), Bottom(), Right(), Top()]

    def _set_up_dirichlet_bcs(self, V, bc_types, bc_values, subdomains):
        """
        Set up Dirichlet boundary conditions for the Poisson PDE problem defined
        on the unit square mesh, where V is the function space.
        """
        dirichlet_bcs = []
        
        for i, bc in enumerate(bc_types):
            if bc.lower() == 'dirichlet':
                dirichlet_bcs.append(dl.DirichletBC(
                    V, bc_values[i], subdomains[i]))

        return dirichlet_bcs

    def _set_up_neumann_bcs(self, V, bc_types, bc_values, subdomains):
        """
        Set up Neumann boundary conditions for the Poisson PDE problem defined
        on the unit square mesh, where V is the function space.
        """

        boundary_markers = dl.MeshFunction(
            'size_t', V.mesh(), V.mesh().topology().dim()-1)
        boundary_markers.set_all(0)
        for i, subdomain in enumerate(subdomains):
            subdomain.mark(boundary_markers, i+1)
        ds = dl.Measure('ds', domain=V.mesh(), subdomain_data=boundary_markers)

        neumann_bcs = []
        for i, bc_type in enumerate(bc_types):
            if bc_type.lower() == 'neumann':
                neumann_bcs.append( lambda m, p: bc_values[i]*p*ds(i+1))

        if neumann_bcs == []:
            return lambda m, p: dl.Constant(0)*p*dl.ds
        else:
            return lambda m, p: sum([nbc(m, p) for nbc in neumann_bcs])
