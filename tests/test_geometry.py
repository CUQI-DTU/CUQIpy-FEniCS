import dolfin as dl
from cuqipy_fenics.geometry import (FEniCSContinuous,
                                    MaternKLExpansion,
                                    StepExpansion)
import numpy as np
import pytest
from scipy.optimize import check_grad

def test_MaternKLExpansion():
    """Test creating a MaternKLExpansion geometry"""
    mesh = dl.UnitSquareMesh(20, 20)
    V = dl.FunctionSpace(mesh, 'CG', 1)
    geometry = FEniCSContinuous(V)
    MaternGeometry = MaternKLExpansion(geometry,
                                     length_scale=.2,
                                     num_terms=128)
    assert (MaternGeometry.num_terms == 128 and np.isclose(
        MaternGeometry.length_scale, .2))


def test_MaternKLExpansion_basis(copy_reference):
    """Test MaternKLExpansion geometry basis building"""

    # Create the MaternKLExpansion geometry
    np.random.seed(0)
    mesh = dl.UnitSquareMesh(20, 20)
    V = dl.FunctionSpace(mesh, 'CG', 1)
    geometry = FEniCSContinuous(V)
    MaternGeometry = MaternKLExpansion(geometry,
                                     length_scale=.2,
                                     num_terms=128,
                                     normalize=False)

    # Build the basis
    MaternGeometry._build_basis()

    # Read the reference eigenvalues and eigenvectors
    samples_orig_file = copy_reference("data/MaternExpansion_basis.npz")
    expected_eig_pairs = np.load(samples_orig_file)
    expected_eig_vec = expected_eig_pairs['expected_eig_vec']
    expected_eig_val = expected_eig_pairs['expected_eig_val']

    # Assert that the eigenvalues match the reference
    assert (np.allclose(MaternGeometry._eig_val, expected_eig_val))

    # Assert that the eigenvectors match the reference, up to sign
    for i in range(MaternGeometry.num_terms):
        assert (np.allclose(MaternGeometry.eig_vec[:, i], expected_eig_vec[:, i]) or np.allclose(
            MaternGeometry.eig_vec[:, i], -expected_eig_vec[:, i]))

    # Assert that the eigenvectors has the correct shape
    assert expected_eig_vec.shape == (441, 128)


@pytest.mark.parametrize("nu, valid",
                         [(0, False), (0.01, True), (-1, False)])
def test_MaternKLExpansion_nu(nu, valid):
    """Test passing nu to the MaternKLExpansion geometry"""
    mesh = dl.UnitSquareMesh(20, 20)
    V = dl.FunctionSpace(mesh, 'CG', 1)
    geometry = FEniCSContinuous(V)

    if valid:
        MaternGeometry = MaternKLExpansion(geometry,
                                         length_scale=0.2,
                                         nu=nu,
                                         num_terms=128)
    else:
        with pytest.raises(ValueError):
            MaternGeometry = MaternKLExpansion(geometry,
                                             length_scale=0.2,
                                             nu=nu,
                                             num_terms=128)

def create_step_expansion_geometry(num_steps_x, num_steps_y):
    """Create a step expansion geometry"""
    mesh = (
        dl.UnitSquareMesh(32, 32)
        if num_steps_y is not None
        else dl.UnitIntervalMesh(32)
    )
    V = dl.FunctionSpace(mesh, "DG", 0)
    G_FEM = FEniCSContinuous(V, labels=["$\\xi_1$", "$\\xi_2$"])
    G_step = StepExpansion(G_FEM, num_steps_x=num_steps_x, num_steps_y=num_steps_y)
    return G_step

@pytest.mark.parametrize(
    "num_steps_x, num_steps_y, expected_vals",
    [
        (8, 8, [0.0, 27.0, 63.0, 59.0, 31.0, 3.0, 24.0, 7.0, 56.0]),
        (1, 8, [0.0, 3.0, 7.0, 7.0, 3.0, 0.0, 3.0, 0.0, 7.0]),
        (4, 8, [0.0, 13.0, 31.0, 29.0, 15.0, 1.0, 12.0, 3.0, 28.0]),
        (1, 1, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (9, None, [0.0, 4.0, 8.0, 4.0, 8.0, 4.0, 0.0, 8.0, 0.0]),
    ],
)
def test_step_expansion_geometry_par2fun_gives_correct_values(
    num_steps_x, num_steps_y, expected_vals
):
    """Test the step expansion geometry par2fun method"""
    G_step = create_step_expansion_geometry(num_steps_x, num_steps_y)

    # create a parameter vector x
    x = (
        np.arange(num_steps_x * num_steps_y)
        if num_steps_y is not None
        else np.arange(num_steps_x)
    )

    # get the function values corresponding to the parameter vector x
    funvals = G_step.par2fun(x)

    # test the function values at some locations and compare with the expected values
    test_locations = [
        (0, 0),
        (0.49, 0.49),
        (1, 1),
        (0.49, 1),
        (1, 0.49),
        (0.49, 0),
        (0, 0.49),
        (1, 0),
        (0, 1),
    ]
    for i, loc in enumerate(test_locations):
        loc = loc if num_steps_y is not None else loc[0]
        assert np.allclose(funvals(loc), expected_vals[i])

def test_step_expansion_gradient_is_correct():
    """Test the gradient of the step expansion geometry"""
    G_step = create_step_expansion_geometry(4, 8)
    V = G_step.function_space
    
    np.random.seed(0)
    # random direction y
    y0 = np.random.randn(G_step.funvec_dim)
    
    # corresponding fenics function
    y0_fun = dl.Function(V)
    y0_fun.vector()[:] = y0

    # objective function
    def f(x):
        return G_step.par2fun(x).vector().get_local().T@ y0.reshape(-1,1)
    
    # objective function gradient
    def fprime(x):
        return G_step.gradient(y0_fun, x)
    
    # random input x (the point which gradient is calculated with respect to)
    x0 = np.random.randn(G_step.par_dim)

    # assert that the gradient is correct
    assert np.allclose(check_grad(f, fprime, x0), 0, atol=1e-5)
