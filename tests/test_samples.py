import pytest
from cuqipy_fenics.geometry import FEniCSContinuous, FEniCSMappedGeometry, MaternKLExpansion
from cuqi.distribution import Gaussian
from cuqipy_fenics.utilities import _compute_stats
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt


@pytest.fixture(params=["G_FE", "G_KL", "G_map"])
def samples(request):
    """ Create Samples object with different geometries"""
    # Define the mesh and the function space
    geometry_type = request.param
    Ns = 5
    mesh = dl.UnitSquareMesh(10, 10)
    V = dl.FunctionSpace(mesh, "CG", 2) 

    # Define the geometries depending on the case
    # FEniCSContinuous geometry
    geom = FEniCSContinuous(V)
    
    if geometry_type == "G_KL" or geometry_type == "G_map":
        # A KL expansion geometry
        num_terms = 10
        length_scale = 0.05
        geom = MaternKLExpansion(geom, length_scale, num_terms)
    
    if geometry_type == "G_map":
        # A mapped geometry applied to the KL expansion geometry
        c_minus = 1
        c_plus = 10
        def heavy_map(func):
            dofs = func.vector().get_local()
            updated_dofs = c_minus*0.5*(1 + np.sign(dofs)) +\
                c_plus*0.5*(1 - np.sign(dofs))
            func.vector().set_local(updated_dofs)
            return func
        geom = FEniCSMappedGeometry(geom, heavy_map)
    
    # Create a distribution and sample the distribution
    x = Gaussian(0, np.ones(geom.par_dim), geometry=geom)
    samples = x.sample(Ns)
    return samples


def test_computing_samples_statistics_is_correct(samples):
    """Test that the mean and variance computed on the samples'
    function representation and the samples' parameter representation
    are correct."""
    
    V = samples.geometry.function_space
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
    mean_helper, var1_helper, var2_helper = _compute_stats(samples)

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


@pytest.mark.parametrize("mesh", [dl.UnitSquareMesh(20, 20), dl.UnitIntervalMesh(20)])  
@pytest.mark.parametrize("basis", ['CG', 'DG'])
@pytest.mark.parametrize("order", [1, 2])
def test_method_plot_envelope_works_only_for_1D_1st_order_Lagrange_space(
        mesh, basis, order):
    """Test that the method plot_envelope used in plot_ci works only for 
    1D 1st order Lagrange space and raises an error otherwise"""

    # define the function space
    space = dl.FunctionSpace(mesh, basis, order)

    # Set the geometry (mapped KL) 
    G_FEM = FEniCSContinuous(space)
    G_KL = MaternKLExpansion(G_FEM, length_scale=0.1, nu=0.75, num_terms=10)

    # define some map
    def prior_map(func):
        dofs = func.vector().get_local()
        updated_dofs = 15*dofs
        func.vector().set_local(updated_dofs)
        return func

    G = FEniCSMappedGeometry(G_KL, map=prior_map)

    # define distribution
    x = Gaussian(0, cov=1, geometry=G)

    # sample the distribution
    samples = x.sample(20)
    exact = x.sample()

    # plot the envelope only if the order is 1 and the basis is CG
    # and the physical dimension is 1. Otherwise, assert that the method
    # plot_envelope raises an error
    if order == 1 and basis == 'CG' and G_FEM.physical_dim == 1:
        # plot_ci which uses _plot_envelope method
        samples.funvals.vector.plot_ci(exact=exact)
    else:
        # assert that the method plot_envelope raises an error
        with pytest.raises(NotImplementedError,
                           match="Envelope plot is only implemented"):
            samples.funvals.vector.plot_ci(exact=exact)