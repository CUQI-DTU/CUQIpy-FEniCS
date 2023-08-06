import dolfin as dl
import cuqipy_fenics
import numpy as np
import pytest
import matplotlib.pyplot as plt
from cuqi.distribution import Gaussian
from cuqi.geometry import Continuous1D
from cuqipy_fenics.testproblem import FEniCSPoisson2D


@pytest.mark.parametrize("bc_types, bc_values, case", [
    (None, None, "valid_1"),
    (['Neumann', 'Dirichlet', 'dirichlet', 'neumann'],
     [-2, lambda x: 0.9*np.sin(x[0])+1, 1, lambda x: x[0]+0.1],
     "valid_2"),
    (['Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet'],
     [0, 3, 0, 1],
     "valid_3"),
    (['Neumann', 'Neumann', 'Neumann', 'neumann'],
     [0, 1, 0, 1],
     "all_neumann_error"),
    (['Neumann', 'Dirichlet'],
     [0, 1],
     "missing_bc_error")])
def test_FEniCSPoisson2D_bc(bc_types, bc_values, case):
    """Test creating a FEniCSPoisson2D testproblem with different boundary conditions"""
    if case[:5] == "valid":
        testproblem = FEniCSPoisson2D((20, 20),
                                      bc_types=bc_types, bc_values=bc_values, source_term=0)

        testproblem.data.plot()
        testproblem.exactData.plot()
        testproblem.exactSolution.plot()

    if case == "all_neumann_error":
        with pytest.raises(ValueError, match=r"All boundary conditions cannot be Neumann"):
            testproblem = FEniCSPoisson2D((20, 20),
                                          bc_types=bc_types, bc_values=bc_values, source_term=0)

    if case == "missing_bc_error":
        with pytest.raises(ValueError, match=r"length of bc_types list should be 4"):
            testproblem = FEniCSPoisson2D((20, 20),
                                          bc_types=bc_types, bc_values=bc_values, source_term=0)

    # Check that the Dirichlet boundary conditions are sat up
    #  correctly by comparing to the exact solution
    if case == "valid_3":
        func = testproblem.exactData.funvals
        assert np.isclose(func(0, 0.5), bc_values[0])
        assert np.isclose(func(0.5, 0), bc_values[1])
        assert np.isclose(func(1, 0.5), bc_values[2])
        assert np.isclose(func(0.5, 1), bc_values[3])

    if case == "valid_2":
        func = testproblem.exactData.funvals
        assert np.isclose(func(0.5, 0), bc_values[1]([0.5, 0]))
        assert np.isclose(func(1, 0.5), bc_values[2])


@pytest.mark.parametrize("source_term, valid", [
    (None, True),
    (3, True),
    (lambda x: x[0], True),
    (dl.Constant(4), True)])
def test_FEniCSPoisson2D_source_term(source_term, valid):
    """Test creating a FEniCSPoisson2D testproblem with different source terms"""
    if valid:
        testproblem = FEniCSPoisson2D((30, 30), source_term=source_term)

        testproblem.data.plot()
        testproblem.exactData.plot()
        testproblem.exactSolution.plot()


@pytest.mark.parametrize(
    "exactSolution, noise_level, field_type, field_params, mapping," +
    "prior, case",
    [(None, 0.02, None, None, None, None, "valid"),
     (None, 0.02, None, None, None, None, "unknown_field"),
     (None, 0.05, None, None, "exponential",
      Gaussian(np.zeros(21*21), 1), "valid"),
     (lambda x: x[0]+x[1]+0.1, 0.02, None, None, "exponential",
      Gaussian(np.zeros(21*21), 1, name='x'), "valid"),
     (None, 0.05, None, None, lambda m: m+1, None, "valid"),
     (None, 0.1, "KL", None, "exponential", None, "valid"),
     (np.random.randn(10), 0.02, "KL",
      {'length_scale': 0.2, 'num_terms': 10}, "exponential", None, "valid")])
def test_FEniCSPoisson2D_setup(exactSolution, noise_level,
                               field_type, field_params, mapping,
                               prior, case):
    """Test creating a FEniCSPoisson2D testproblem with different 
    parametrization, exact solution and noise level"""
    # set the random seed
    np.random.seed(0)

    if case == "valid":
        testproblem = FEniCSPoisson2D(
            (20, 20),
            exactSolution=exactSolution,
            noise_level=noise_level,
            field_type=field_type,
            field_params=field_params,
            mapping=mapping,
            prior=prior)

        testproblem.data.plot()
        testproblem.exactData.plot()
        testproblem.exactSolution.plot()
        testproblem.UQ(Ns=20, percent=90)

    if case == "unknown_field":
        with pytest.raises(ValueError, match=r"Unknown field type"):
            testproblem = FEniCSPoisson2D(
                (20, 20),
                exactSolution=exactSolution,
                noise_level=noise_level,
                field_type=Continuous1D(21*21),
                field_params=field_params,
                mapping=mapping,
                prior=prior)

    # Check the noise level is correct
    if case == "valid":
        noise_norm =\
            np.linalg.norm(testproblem.data.funvals.vector().get_local()
                           - testproblem.exactData.funvals.vector().get_local())
        exact_norm = np.linalg.norm(
            testproblem.exactData.funvals.vector().get_local())
        assert np.isclose(noise_norm/exact_norm, noise_level)
