import dolfin as dl
from cuqipy_fenics.geometry import FEniCSContinuous, MaternKLExpansion
import numpy as np
import pytest


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
