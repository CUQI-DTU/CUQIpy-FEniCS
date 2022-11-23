import dolfin as dl
import cuqi
import cuqipy_fenics
import numpy as np
import pytest
import time


def test_model_input():
    """Test passing different data structures for PDEModel input"""
    model = cuqipy_fenics.testproblem.FEniCSDiffusion1D().model
    V = model.pde.parameter_function_space

    # Test passing a CUQIarray containing parameters
    u = dl.Function(V)
    u_CUQIarray = cuqi.samples.CUQIarray(u.vector().get_local(), is_par=True, geometry=model.domain_geometry)
    y = model(u_CUQIarray)

    # Test passing parameters as a numpy array
    u = dl.Function(V)
    u_numpy = u.vector().get_local()
    y = model(u_numpy, is_par=True)

    # Test passing a CUQIarray containing dolfin function
    u = dl.Function(V)
    u_CUQIarray = cuqi.samples.CUQIarray(u, is_par=False, geometry=model.domain_geometry)
    y = model(u_CUQIarray)

    # Test passing a CUQIarray containing dolfin function wrapped in np.array
    # This is not recommended, but should work
    u = dl.Function(V)
    u_CUQIarray = cuqi.samples.CUQIarray(np.array(u, dtype='O'), is_par=False, geometry=model.domain_geometry)
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
        poisson.bc,
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
        poisson.bc,
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
            poisson.bc,
            reuse_assembled=True)

    # Create a PDE object
    PDE = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson.lhs_form, poisson.rhs_form),
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bc,
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
        poisson.bc)

    # Solve the PDE
    PDE_with_full_form.assemble(m)
    u1, info = PDE_with_full_form.solve()

    # Create a PDE object with lhs and rhs forms
    PDE_with_lhs_rhs_forms = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson.lhs_form, poisson.rhs_form),
        poisson.mesh,
        poisson.solution_function_space,
        poisson.parameter_function_space,
        poisson.bc)

    # Solve the PDE
    PDE_with_lhs_rhs_forms.assemble(m)
    u2, info = PDE_with_lhs_rhs_forms.solve()

    # Check that the solutions are the same
    assert np.allclose(u1.vector().get_local(), u2.vector().get_local())


def test_with_updated_rhs(copy_reference):
    """ Regression test for using the method with_updated_rhs and sharing the
    factored lhs matrix"""

    # Set up first poisson problem
    poisson1 = Poisson()
    poisson1.mesh = dl.UnitSquareMesh(40, 40)
    poisson1.f = dl.Constant(1.0)

    # Set up second poisson problem
    poisson2 = Poisson()
    poisson2.mesh = poisson1.mesh
    poisson2.f = dl.Expression("sin(2*x[0]*pi)*sin(2*x[1]*pi)", degree=1)

    # Set up boundary function (where the Dirichlet boundary conditions are applied)
    def u_boundary(x, on_boundary):
        return on_boundary and\
            (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

    poisson1.bc = dl.DirichletBC(poisson1.solution_function_space,
                                 dl.Constant(0.0), u_boundary)
    poisson2.bc = poisson1.bc

    # Create two PDE objects with different rhs terms
    PDE1 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson1.lhs_form, poisson1.rhs_form),
        poisson1.mesh,
        parameter_function_space=poisson1.parameter_function_space,
        solution_function_space=poisson1.solution_function_space,
        dirichlet_bc=poisson1.bc,
        observation_operator=None,
        reuse_assembled=False)

    PDE2 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(
        (poisson2.lhs_form, poisson2.rhs_form),
        poisson2.mesh,
        parameter_function_space=poisson2.parameter_function_space,
        solution_function_space=poisson2.solution_function_space,
        dirichlet_bc=poisson2.bc,
        observation_operator=None,
        reuse_assembled=False)

    # Create domain geometry
    fenics_continuous_geo = cuqipy_fenics.geometry.FEniCSContinuous(
        poisson1.parameter_function_space)
    domain_geometry = cuqipy_fenics.geometry.MaternExpansion(
        fenics_continuous_geo, length_scale=.1, num_terms=32)

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
    exact_solution = cuqi.samples.CUQIarray(
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
    Ns = 10
    np.random.seed(0)
    sampler = cuqi.sampler.MetropolisHastings(cuqi_posterior)
    t0 = time.time()
    samples1 = sampler.sample_adapt(Ns, Nb=10)
    t1 = time.time()
    t_no_reuse = t1-t0
    samples1.geometry = domain_geometry

    # Set PDE2 to be a shallow copy of PDE1 but with different rhs
    PDE1.reuse_assembled = True
    PDE2 = PDE1.with_updated_rhs(poisson2.rhs_form)
    cuqi_model2.pde = PDE2

    # Sample the posterior again (Case 2: reuse of assembled operators)
    np.random.seed(0)
    sampler = cuqi.sampler.MetropolisHastings(cuqi_posterior)
    t0 = time.time()
    samples2 = sampler.sample_adapt(Ns, Nb=10)
    t1 = time.time()
    t_reuse = t1-t0

    # Check that the samples are the same
    assert np.allclose(samples1.samples, samples2.samples)

    # Check that the samples matches the ones generated
    # before updating the library code to add the reuse_assembled functionality
    samples_orig_file =\
        copy_reference(
            "data/samples_before_adding_reuse_assembled_feature.npz")
    assert np.allclose(samples1.samples, np.load(
        samples_orig_file)["samples_orig"][:, :Ns])

    # Check that reusing factorization and with_updated_rhs is faster
    assert t_reuse < 0.7*t_no_reuse


class Poisson:
    """Define the variational PDE problem for the Poisson equation in two
    ways: as a full form, and as lhs and rhs forms"""

    def __init__(self):

        # Create the mesh and define function spaces for the solution and the
        # parameter
        self.mesh = dl.UnitIntervalMesh(10)

        # Define the boundary condition
        self.bc_value = dl.Constant(0.0)

        # the source term
        self.f = dl.Expression('x[0]', degree=1)

    @property
    def form(self):
        return lambda m, u, v: dl.exp(m)*dl.inner(dl.grad(u), dl.grad(v))*dl.dx\
            + self.f*v*dl.dx

    @property
    def lhs_form(self):
        return lambda m, u, v: dl.exp(m)*dl.inner(dl.grad(u), dl.grad(v))*dl.dx

    @property
    def rhs_form(self):
        return lambda m, v: -self.f*v*dl.dx

    @property
    def solution_function_space(self):
        return dl.FunctionSpace(self.mesh, "CG", 2)

    @property
    def parameter_function_space(self):
        return dl.FunctionSpace(self.mesh, "CG", 1)

    @property
    def bc(self):
        if not hasattr(self, "_bc") or self._bc is None:
            self._bc = dl.DirichletBC(self.solution_function_space,
                                      self.bc_value, "on_boundary")
        return self._bc

    @bc.setter
    def bc(self, bc):
        self._bc = bc
