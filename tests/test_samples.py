import pytest
from cuqipy_fenics.geometry import FEniCSContinuous, FEniCSMappedGeometry, MaternKLExpansion
from cuqi.distribution import Gaussian
from cuqipy_fenics.utilities import compute_stats
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

@pytest.mark.parametrize("case", ["G_FEM", "G_KL", "G_map"])
def test_samples_statistics_par_fun(case):

    # Define the mesh and the function space
    mesh = dl.UnitSquareMesh(10, 10)
    V = dl.FunctionSpace(mesh, "CG", 2) 

    # Define the geometries depending on the case
    # FEniCSContinuous geometry
    geom = FEniCSContinuous(V)
    
    if case == "G_KL" or case == "G_map":
        # A KL expansion geometry
        num_terms = 10
        length_scale = 0.05
        geom = MaternKLExpansion(geom, length_scale, num_terms)
    
    if case == "G_map":
        # A mapped geometry applied to the KL expansion geometry
        c_minus = 1
        c_plus = 10
        def heavy_map(func):
            dofs = func.vector().get_local()
            updated_dofs = c_minus*0.5*(1 + np.sign(dofs)) + c_plus*0.5*(1 - np.sign(dofs))
            func.vector().set_local(updated_dofs)
            return func
        geom = FEniCSMappedGeometry(geom, heavy_map)
    
    
    # Create a distribution and sample the distribution
    x = Gaussian(0, np.ones(geom.par_dim), geometry=geom)
    samples = x.sample(5)
    samples_funvals = samples.funvals
    
    # Compute mean and variance on the **parameter** space
    mean = samples.mean()
    var = samples.variance()

    assert np.allclose(mean, samples.samples.mean(axis=1))
    assert np.allclose(var, samples.samples.var(axis=1))

    # Compute mean and variance on the **function** space (this will generate an
    # error because the samples are not converted to vector form)
    with pytest.raises(Exception, match="cuqi.samples._samples added message:"):
        samples_funvals.mean()

    with pytest.raises(Exception, match="cuqi.samples._samples added message:"):
        samples_funvals.variance()

    # Compute mean and variance on the **function** space after converting the
    # samples to vector form
    mean_funvals = samples_funvals.vector.mean()
    var_funvals = samples_funvals.vector.variance()

    # Compute mean and variance on the **function** space using helper function
    mean_helper, var1_helper, var2_helper = compute_stats(samples)

    # var1_helper is computed on a higher order function space, so we need to
    # evaluate it at the DOFs of the original function space V to compare with
    # var_funvals
    var1_helper_V_DOF = np.empty(V.dim())
    for i in range(V.dim()):
        var1_helper_V_DOF[i] = var1_helper(V.tabulate_dof_coordinates()[i,:])
 
    # Assert that the mean and variance computed on the function space are 
    # correct
    assert np.allclose(mean_funvals, mean_helper.vector().get_local())
    assert np.allclose(var_funvals, var2_helper.vector().get_local())
    assert np.allclose(var_funvals, var1_helper_V_DOF)
