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
