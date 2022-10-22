import dolfin as dl
import cuqipy_fenics
import numpy as np

def test_MaternExpansion():
    """Test creating a MaternExpansion geometry"""
    mesh = dl.UnitSquareMesh(20,20)
    V = dl.FunctionSpace(mesh, 'CG', 1)
    geometry = cuqipy_fenics.geometry.FEniCSContinuous(V)
    MaternGeometry = cuqipy_fenics.geometry.MaternExpansion(geometry, 
                                    length_scale = .2,
                                    num_terms=128)
    assert(MaternGeometry.num_terms == 128 and np.isclose(MaternGeometry.length_scale, .2))


def test_MaternExpansion_basis(copy_reference):
    """Test MaternExpansion geometry basis building"""

    # Create the MaternExpansion geometry
    np.random.seed(0)
    mesh = dl.UnitSquareMesh(20, 20)
    V = dl.FunctionSpace(mesh, 'CG', 1)
    geometry = cuqipy_fenics.geometry.FEniCSContinuous(V)
    MaternGeometry = cuqipy_fenics.geometry.MaternExpansion(geometry,
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
