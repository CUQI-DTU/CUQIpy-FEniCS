import dolfin as dl
import cuqi
import cuqipy_fenics
import numpy as np
import pytest


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
