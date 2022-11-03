#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
from cuqipy_fenics.geometry import MaternExpansion, FEniCSContinuous
from cuqi.distribution import Gaussian
import dolfin as dl

mesh = dl.UnitSquareMesh(20,20)
V = dl.FunctionSpace(mesh, 'CG', 1)
geometry = FEniCSContinuous(V)
MaternGeometry = MaternExpansion(geometry, 
                                length_scale = .2,
                                num_terms=128)

MaternField = Gaussian(mean=np.zeros(MaternGeometry.par_dim),
                       cov=np.eye(MaternGeometry.par_dim),
                       geometry=MaternGeometry)

samples = MaternField.sample()
samples.plot()

# View the first 10 eigenvectors
for i in range(10):
    plt.figure()
    geometry.plot(MaternGeometry.eig_vec[:,i]) 
    plt.show()
    plt.close('all')
