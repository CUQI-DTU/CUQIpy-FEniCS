import dolfin as dl
import cuqipy_fenics

def test_dolfin_mesh():
    """Test creating a MaternExpansion geometry"""
    mesh = dl.UnitSquareMesh(20,20)
    V = dl.FunctionSpace(mesh, 'CG', 1)
    geometry = cuqipy_fenics.FEniCSContinuous(V)
    MaternGeometry = cuqipy_fenics.MaternExpansion(geometry, 
                                    length_scale = .2,
                                    num_terms=128)
    assert(True)
