import dolfin as dl
import cuqi
import cuqipy_fenics
import numpy as np
import pytest
import time
import sys
from scipy import optimize
ufl = cuqipy_fenics.utilities._import_ufl()

def test_model_input():
    """Test passing different data structures for PDEModel input"""
    model = cuqipy_fenics.testproblem.FEniCSDiffusion1D().model
    V = model.pde.parameter_function_space

    # Test passing a CUQIarray containing parameters
    u = dl.Function(V)
    u_CUQIarray = cuqi.array.CUQIarray(u.vector().get_local(), is_par=True, geometry=model.domain_geometry)
    y = model(u_CUQIarray)

    # Test passing parameters as a numpy array
    u = dl.Function(V)
    u_numpy = u.vector().get_local()
    y = model(u_numpy, is_par=True)

    # Test passing a CUQIarray containing dolfin function
    u = dl.Function(V)
    u_CUQIarray = cuqi.array.CUQIarray(u, is_par=False, geometry=model.domain_geometry)
    y = model(u_CUQIarray)

    # Test passing a CUQIarray containing dolfin function wrapped in np.array
    # This is not recommended, but should work
    u = dl.Function(V)
    u_CUQIarray = cuqi.array.CUQIarray(np.array(u, dtype='O'), is_par=False, geometry=model.domain_geometry)
    y = model(u_CUQIarray)  

    # Test passing a dolfin function (should fail)
    u = dl.Function(V)
    with pytest.raises(AttributeError):
        y = model(u)


def test_solver_choice():
    """Test passing different solvers to PDEModel"""
    # Create the variational problem
    poisson = Poisson()

    # Set up model parameters
    m = dl.Function(poisson.parameter_function_space)
    m.vector()[:] = 1.0

    # Create a PDE object with Krylov solver
    PDE_with_solver1 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        poisson.form,
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs,
        linalg_solve=dl.KrylovSolver())

    # Solve the PDE
    PDE_with_solver1.assemble(m)
    u1, info = PDE_with_solver1.solve()

    # Create a PDE object with direct LU solver
    PDE_with_solver2 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        poisson.form,
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs,
        linalg_solve=dl.LUSolver())

    # Solve the PDE
    PDE_with_solver2.assemble(m)
    u2, info = PDE_with_solver2.solve()

    # Check that the solutions are the same
    assert np.allclose(u1.vector().get_local(), u2.vector().get_local())


def test_reuse_assembled():
    """Test that reusing the assembled and factored lhs gives the same solution
     and a better performance"""
    # Create the variational problem
    poisson = Poisson()
    poisson.mesh = dl.UnitSquareMesh(50, 50)

    # Set up model parameters
    m = dl.Function(poisson.parameter_function_space)
    m.vector()[:] = 1.0

    # Create a PDE object, generate an error if the full form is provided
    # but reuse_assembled is True
    with pytest.raises(ValueError):
        PDE = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
            poisson.form,
            poisson.mesh,
            poisson.solution_function_space,
            poisson.parameter_function_space,
            poisson.bcs,
            reuse_assembled=True)

    # Create a PDE object
    PDE = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson.lhs_form, poisson.rhs_form),
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs,
        reuse_assembled=True)

    # Solve the PDE
    PDE.assemble(m)
    u1, info = PDE.solve()

    # update the parameter and solve again
    sin_func = dl.Expression("sin(2*pi*x[0])", degree=1)
    m.interpolate(sin_func)
    t0 = time.time()
    PDE.assemble(m)
    u1, info = PDE.solve()
    t1 = time.time()
    t_first = t1 - t0

    # Solve the PDE again
    t0 = time.time()
    PDE.assemble(m)
    u2, info = PDE.solve()
    t1 = time.time()
    t_reuse_factors = t1 - t0

    # Check that the solutions are the same
    assert np.allclose(u1.vector().get_local(), u2.vector().get_local())

    # Check that using the reuse_assembled option is faster
    assert t_reuse_factors < 0.25*t_first


def test_form():
    """Test creating PDEModel with full form, and with lhs and rhs forms"""
    # Create the variational problem
    poisson = Poisson()

    # Set up model parameters
    m = dl.Function(poisson.parameter_function_space)
    m.vector()[:] = 1.0

    # Create a PDE object with full form
    PDE_with_full_form = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        poisson.form,
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs)

    # Solve the PDE
    PDE_with_full_form.assemble(m)
    u1, info = PDE_with_full_form.solve()

    # Create a PDE object with lhs and rhs forms
    PDE_with_lhs_rhs_forms = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson.lhs_form, poisson.rhs_form),
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs)

    # Solve the PDE
    PDE_with_lhs_rhs_forms.assemble(m)
    u2, info = PDE_with_lhs_rhs_forms.solve()

    # Check that the solutions are the same
    assert np.allclose(u1.vector().get_local(), u2.vector().get_local())

@pytest.mark.parametrize("case", [1, 2])
def test_with_updated_rhs(copy_reference, case):
    """Regression test for using the method with_updated_rhs and sharing the
    factored lhs matrix. Case 1: compare the solution reusing the factorization
    and without reusing the factorization. Case 2: compare the solution with
    results from the previous version of the code."""

    # Skip case 2 test if the operating system is not Mac OS X
    if  case == 2 and not sys.platform.startswith('darwin'):
        pytest.skip("Test on MAC OS only")

    # Set up first poisson problem
    poisson1 = Poisson()
    poisson1.mesh = dl.UnitSquareMesh(40, 40)
    poisson1.source_term = dl.Constant(1.0)

    # Set up second poisson problem
    poisson2 = Poisson()
    poisson2.mesh = poisson1.mesh
    poisson2.source_term = dl.Expression("sin(2*x[0]*pi)*sin(2*x[1]*pi)", degree=1)

    # Set up boundary function (where the Dirichlet boundary conditions are applied)
    def u_boundary(x, on_boundary):
        return on_boundary and\
            (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

    poisson1.bcs = dl.DirichletBC(poisson1.solution_function_space,
                                 dl.Constant(0.0), u_boundary)
    poisson2.bcs = poisson1.bcs

    # Create two PDE objects with different rhs terms
    PDE1 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson1.lhs_form, poisson1.rhs_form),
        poisson1.mesh,
        parameter_function_space=poisson1.parameter_function_space,
        solution_function_space=poisson1.solution_function_space,
        dirichlet_bcs=poisson1.bcs,
        observation_operator=None,
        reuse_assembled=False)

    PDE2 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson2.lhs_form, poisson2.rhs_form),
        poisson2.mesh,
        parameter_function_space=poisson2.parameter_function_space,
        solution_function_space=poisson2.solution_function_space,
        dirichlet_bcs=poisson2.bcs,
        observation_operator=None,
        reuse_assembled=False)

    # Create domain geometry
    fenics_continuous_geo = cuqipy_fenics.geometry.FEniCSContinuous(
        poisson1.parameter_function_space)
    domain_geometry = cuqipy_fenics.geometry.MaternKLExpansion(
        fenics_continuous_geo, length_scale=.1, num_terms=5)

    # Create range geometry
    range_geometry = cuqipy_fenics.geometry.FEniCSContinuous(
        poisson1.solution_function_space)

    # Create cuqi forward model (two models corresponding to two PDE objects)
    cuqi_model1 = cuqi.model.PDEModel(
        PDE1, domain_geometry=domain_geometry, range_geometry=range_geometry)

    cuqi_model2 = cuqi.model.PDEModel(
        PDE2, domain_geometry=domain_geometry, range_geometry=range_geometry)

    # Create the prior
    x = cuqi.distribution.Gaussian(mean=np.zeros(cuqi_model1.domain_dim),
                                   cov=1, geometry=domain_geometry)

    # Create exact solution and data
    np.random.seed(0)
    exact_solution = cuqi.array.CUQIarray(
        np.random.randn(domain_geometry.par_dim),
        is_par=True, geometry=domain_geometry)

    data1 = cuqi_model1(exact_solution)
    data2 = cuqi_model2(exact_solution)

    # Create likelihood 1
    y1 = cuqi.distribution.Gaussian(mean=cuqi_model1(x),
                                    cov=np.ones(cuqi_model1.range_dim)*.01**2)
    y1 = y1(y1=data1)

    # Create likelihood 2
    y2 = cuqi.distribution.Gaussian(mean=cuqi_model2(x),
                                    cov=np.ones(cuqi_model2.range_dim)*.01**2)
    y2 = y2(y2=data2)

    # Create posterior
    cuqi_posterior = cuqi.distribution.JointDistribution(
        y1, y2, x)._as_stacked()

    # Solve the Bayesian problem
    # Sample the posterior (Case 1: no reuse of assembled operators)
    Ns = 20
    np.random.seed(0)
    sampler = cuqi.sampler.MH(cuqi_posterior)
    t0 = time.time()
    samples1 = sampler.sample_adapt(Ns, Nb=10)
    t1 = time.time()
    t_no_reuse = t1-t0
    samples1.geometry = domain_geometry

    if case == 1:
        # Set PDE2 to be a shallow copy of PDE1 but with different rhs
        PDE1.reuse_assembled = True
        PDE2 = PDE1.with_updated_rhs(poisson2.rhs_form)
        cuqi_model2.pde = PDE2

        # Sample the posterior again (Case 2: reuse of assembled operators)
        np.random.seed(0)
        sampler = cuqi.sampler.MH(cuqi_posterior)
        t0 = time.time()
        samples2 = sampler.sample_adapt(Ns, Nb=10)
        t1 = time.time()
        t_reuse = t1-t0

        # Check that the samples are the same
        assert np.allclose(samples1.samples, samples2.samples)
        
        # Check that reusing factorization and with_updated_rhs is faster
        assert t_reuse < 0.7*t_no_reuse

    elif case == 2:
        # Check that the samples matches the ones generated
        # before updating the library code to add the reuse_assembled functionality
        samples_orig = np.load(
            copy_reference('data/samples_test_with_rhs_write_from_pytests'+\
            '_e349834fc189e1e1da6a962cdfe449f94486824e.npz'))["samples1"]
        assert np.allclose(samples1.samples[1, :], samples_orig[1, :])


class Poisson:
    """Define the variational PDE problem for the Poisson equation in two
    ways: as a full form, and as lhs and rhs forms"""

    def __init__(self, mesh):

        # Set the mesh
        self._mesh =mesh

        # Define the boundary condition
        self.bc_value = dl.Constant(0.0)

        # The source term
        self._source_term = dl.Expression('x[0]', degree=1)

        # Solution function space
        self._solution_function_space = dl.FunctionSpace(self.mesh, "Lagrange", 2)

        # Parameter function space
        self._parameter_function_space = dl.FunctionSpace(self.mesh, "Lagrange", 1)

    @property
    def mesh(self):
        return self._mesh

    @property
    def form(self):
        return lambda m, u, v:\
            ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx\
            + self._source_term*v*ufl.dx

    @property
    def lhs_form(self):
        return lambda m, u, v:\
            ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

    @property
    def rhs_form(self):
        return lambda m, v: -self.source_term*v*ufl.dx

    @property
    def solution_function_space(self):
        return self._solution_function_space

    @property
    def parameter_function_space(self):
        return self._parameter_function_space

    @property
    def bcs(self):
        return dl.DirichletBC(
            self.solution_function_space, self.bc_value, "on_boundary"
        )


def test_observation_operator_setter():
    """Test that the observation setter works as expected"""
    # Create the variational problem
    poisson = Poisson()
    poisson.mesh = dl.UnitSquareMesh(50, 50)

    # Set up model parameters
    m = dl.Function(poisson.parameter_function_space)
    m.vector()[:] = 1.0

    # Create the observation operator
    observation_operator = lambda m, u: u.vector().get_local()[200:205] 

    # Create a PDE object (set observation_operator through setter)
    PDE1 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson.lhs_form, poisson.rhs_form),
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs,
        observation_operator=None)
    PDE1.observation_operator = observation_operator

    # Create a PDE object (pass observation_operator as argument)  
    PDE2 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson.lhs_form, poisson.rhs_form),
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs,
        observation_operator=observation_operator)

    # Solve the first PDE
    PDE1.assemble(m)
    u1, info = PDE1.solve()
    u1_obs = PDE1.observe(u1)

    # Solve the second PDE
    PDE2.assemble(m)
    u2, info = PDE2.solve()
    u2_obs = PDE2.observe(u2)

    # Apply the observation operator directly
    u3_obs = observation_operator(None, u2)

    # Check that the solutions are the same
    assert np.allclose(u1_obs, u2_obs) and np.allclose(u2_obs, u3_obs)
    assert len(u1_obs) == 5

def test_gradient_poisson():
    """Test the gradient of the Poisson PDEModel is computed correctly"""
    # Set random seed
    np.random.seed(0)

    # Create the variational problem
    poisson = Poisson(dl.UnitIntervalMesh(40))

    # Set up model parameters
    m = dl.Function(poisson.parameter_function_space)
    m.vector()[:] = 1.0

    # Create a PDE object
    PDE = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        poisson.form,
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs,
        poisson.bcs)

    # Create a PDE model
    PDE_model = cuqi.model.PDEModel(
        PDE,
        domain_geometry=cuqipy_fenics.geometry.FEniCSContinuous(
            poisson.parameter_function_space
        ),
        range_geometry=cuqipy_fenics.geometry.FEniCSContinuous(
            poisson.solution_function_space
        ),
    )

    # Create a prior distribution
    m_prior = cuqi.distribution.Gaussian(
        mean=np.zeros(PDE_model.domain_dim), cov=1, geometry=PDE_model.domain_geometry
    )

    # Create a likelihood
    y = cuqi.distribution.Gaussian(
        mean=PDE_model(m_prior), cov=np.ones(PDE_model.range_dim)*.1**2, geometry=PDE_model.range_geometry)
    y = y(y=PDE_model(m.vector().get_local()))

    # Evaluate the adjoint based gradient at value m2
    m2 = dl.Function(poisson.parameter_function_space)
    m2.vector()[:] = np.random.randn(PDE_model.domain_dim)
    adjoint_grad = y.gradient(m_prior=m2.vector().get_local())

    # Compute the FD gradient
    step = 1e-7   # finite diff step
    FD_grad = optimize.approx_fprime(m2.vector().get_local(), y.logd, step)

    # Check that the adjoint gradient and FD gradient are close
    assert np.allclose(adjoint_grad, FD_grad, rtol=1e-1),\
        f"Adjoint gradient: {adjoint_grad}, FD gradient: {FD_grad}"
    assert np.linalg.norm(adjoint_grad-FD_grad)/ np.linalg.norm(FD_grad) < 1e-2

class PoissonMultipleInputs:
    """Define the variational PDE problem for the Poisson equation with multiple
     unknowns in two ways: as a full form, and as lhs and rhs forms"""

    def __init__(self):

        # Create the mesh and define function spaces for the solution and the
        # parameter
        self.mesh = dl.UnitIntervalMesh(10)

        # Define the boundary condition
        self.bc_value = dl.Constant(0.0)

        # Set the solution function space
        self._solution_function_space =  dl.FunctionSpace(self.mesh, "Lagrange", 2)

        # Set the parameter function space
        self._parameter_function_space = [
            dl.FunctionSpace(self.mesh, "Lagrange", 1),
            dl.FunctionSpace(self.mesh, "Lagrange", 1)]

    @property
    def form(self):
        return lambda m, source_term, u, v:\
            ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx\
            + source_term*v*ufl.dx

    @property
    def lhs_form(self):
        return lambda m, source_term, u, v:\
            ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

    @property
    def rhs_form(self):
        return lambda m, source_term, v: -source_term*v*ufl.dx

    @property
    def solution_function_space(self):
        return self._solution_function_space

    @property
    def parameter_function_space(self):
        return self._parameter_function_space

    @property
    def bcs(self):
        if not hasattr(self, "_bcs") or self._bcs is None:
            self._bcs = dl.DirichletBC(self.solution_function_space,
                                      self.bc_value, "on_boundary")
        return self._bcs

    @bcs.setter
    def bcs(self, bcs):
        self._bcs = bcs

def test_form_multiple_inputs():
    """Test creating PDEModel with full form, and with lhs and rhs forms with multiple inputs"""
    # Create the variational problem
    poisson = PoissonMultipleInputs()

    # Set up model parameters
    m = dl.Function(poisson.parameter_function_space[0])
    m.vector()[:] = 1.0
    source_term = dl.Function(poisson.parameter_function_space[1])
    source_term.interpolate(dl.Expression('x[0]', degree=1))

    # Create a PDE object with full form
    PDE_with_full_form = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        poisson.form,
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs)

    # Solve the PDE
    PDE_with_full_form.assemble(m, source_term)
    PDE_with_full_form.assemble(m=m, source_term=source_term)    
    u1, info = PDE_with_full_form.solve()

    # Create a PDE object with lhs and rhs forms
    PDE_with_lhs_rhs_forms = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson.lhs_form, poisson.rhs_form),
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs)
    # Solve the PDE
    PDE_with_lhs_rhs_forms.assemble(m, source_term)
    u2, info = PDE_with_lhs_rhs_forms.solve()

    # Check that the solutions are the same
    assert np.allclose(u1.vector().get_local(), u2.vector().get_local())

    # Check passing wrong parameter_function_space raises an error
    with pytest.raises(ValueError, match= r"assemble input is specified by keywords arguments \['m2', 'source_term'\] that does not match the non_default_args of assemble \['m', 'source_term'\]"):
        PDE_with_full_form.assemble(m2=m, source_term=source_term)

def test_gradient_poisson_multiple_inputs():
    """Test the gradient of the Poisson PDEModel with multiple inputs is computed correctly"""
    # Set random seed
    np.random.seed(0)

    # Create the variational problem
    poisson = PoissonMultipleInputs()

    # Set up model parameters
    m = dl.Function(poisson.parameter_function_space[0])
    m.vector()[:] = 1.0
    source_term = dl.Function(poisson.parameter_function_space[1])
    source_term.interpolate(dl.Expression('x[0]', degree=1))

    # Create a PDE object
    PDE = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        poisson.form,
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bcs)

    # Create a PDE model
    PDE_model = cuqi.model.PDEModel(
        PDE,
        domain_geometry=cuqipy_fenics.geometry.FEniCSContinuous(
            poisson.parameter_function_space[0]
        ),
        range_geometry=cuqipy_fenics.geometry.FEniCSContinuous(
            poisson.solution_function_space
        ),
    )

    # Create a prior distribution
    m_prior = cuqi.distribution.Gaussian(
        mean=np.zeros(PDE_model.domain_dim), cov=1, geometry=PDE_model.domain_geometry
    )

    # Create a likelihood
    y = cuqi.distribution.Gaussian(
        mean=PDE_model(m_prior, source_term), cov=np.ones(PDE_model.range_dim)*.1**2, geometry=PDE_model.range_geometry)
    y = y(y=PDE_model(m.vector().get_local(), source_term))

    # Evaluate the adjoint based gradient at value m2
    m2 = dl.Function(poisson.parameter_function_space[0])
    m2.vector()[:] = np.random.randn(PDE_model.domain_dim)
    adjoint_grad = y.gradient(m_prior=m2.vector().get_local())

    # Compute the FD gradient
    step = 1e-7   # finite diff step
    FD_grad = optimize.approx_fprime(m2.vector().get_local(), y.logd, step)

    # Check that the adjoint gradient and FD gradient are close
    assert np.allclose(adjoint_grad, FD_grad, rtol=1e-1),\
        f"Adjoint gradient: {adjoint_grad}, FD gradient: {FD_grad}"
    assert np.linalg.norm(adjoint_grad-FD_grad)/ np.linalg.norm(FD_grad) < 1e-2
