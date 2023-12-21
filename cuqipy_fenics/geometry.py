from cuqi.geometry import Geometry, MappedGeometry, _WrappedGeometry, Continuous1D
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
from .utilities import _import_ufl
ufl = _import_ufl()

__all__ = [
    'FEniCSContinuous',
    'FEniCSMappedGeometry',
    'MaternKLExpansion'
]

class FEniCSContinuous(Geometry):

    def __init__(self, function_space, labels = ['x', 'y']):
        self.function_space = function_space
        if self.physical_dim >2:
            raise NotImplementedError("'FEniCSContinuous' object does not support 3D meshes yet. 'mesh' needs to be a 1D or 2D mesh.")
        self.labels = labels

    @property
    def physical_dim(self):
        return self.function_space.mesh().geometry().dim()  

    @property
    def mesh(self):
        return self.function_space.mesh()

    @property
    def par_shape(self):
        return (self.function_space.dim(),)
    
    @property
    def funvec_shape(self):
        """The shape of the geometry (shape of the vector representation of the
        function value)."""
        return (self.function_space.dim(),)
    
    def par2fun(self, par):
        """The parameter to function map used to map parameters to function values in e.g. plotting."""
        par = self._process_values(par)
        Ns = par.shape[-1]
        fun_list = []
        for idx in range(Ns):
            fun = dl.Function(self.function_space)
            fun.vector().zero()
            fun.vector().set_local(par[...,idx])
            fun_list.append(fun)

        if len(fun_list) == 1:
            return fun_list[0]
        else:
            return fun_list

    def fun2par(self, fun):
        """ Maps the function values (FEniCS object) to the corresponding parameters (ndarray)."""
        return fun.vector().get_local()

    def fun2vec(self, fun):
        """ Maps the function value (FEniCS object) to the corresponding vector
        representation of the function (ndarray of the function DOF values)."""
        return self.fun2par(fun)
    
    def vec2fun(self, funvec):
        """ Maps the vector representation of the function (ndarray of the
        function DOF values) to the function value (FEniCS object)."""
        return self.par2fun(funvec)
    
    def gradient(self, direction, wrt=None, is_direction_par=False, is_wrt_par=True):
        """ Computes the gradient of the par2fun map with respect to the parameters in the direction `direction` evaluated at the point `wrt`"""
        if is_direction_par:
            return direction
        else:
            return self.fun2par(direction)

    def _plot(self,values,subplots=True, **kwargs):
        """
        Overrides :meth:`cuqi.geometry.Geometry.plot`. See :meth:`cuqi.geometry.Geometry.plot` for description  and definition of the parameter `values`.
        
        Parameters
        -----------
        kwargs : keyword arguments
            keyword arguments which the function :meth:`dolfin.plot` normally takes.
        """
        # If plotting one parameter/function is required:
        if isinstance(values, dl.function.function.Function)\
           or (hasattr(values,'shape') and len(values.shape) == 1):
            Ns = 1
            values = [values]
        # If plotting multiple parameters/functions is required:
        elif hasattr(values,'__len__'): 
            Ns = len(values)
        subplot_ids = self._create_subplot_list(Ns,subplots=subplots)

        ims = []
        for rows,cols,subplot_id in subplot_ids:
            fun = values[subplot_id-1]
            if subplots:
                plt.subplot(rows,cols,subplot_id); 
            ims.append(dl.plot(fun, **kwargs))

        self._plot_config(subplots) 
        return ims

    def _process_values(self, values):
        if isinstance(values, dl.function.function.Function):
            return [values]

        elif len(values.shape) == 1:
            values = values[..., np.newaxis]
        
        return values
    
    def _plot_config(self, subplot):
        if self.labels is not None:
            if subplot == False:
                plt.gca().set_xlabel(self.labels[0])
                if self.physical_dim == 2: plt.gca().set_ylabel(self.labels[1]) 
            else:
                for i, axis in enumerate(plt.gcf().axes):
                    axis.set_xlabel(self.labels[0])
                    if self.physical_dim == 2: axis.set_ylabel(self.labels[1])

    def _plot_envelope(self, lo_values, up_values, **kwargs):
        """Method to plot the envelope of the lower and upper bounds of the
        function values. This method is only implemented for Lagrange finite
        element space of order 1 and 1D meshes."""

        # Check the conditions above:
        ufl_element = self.function_space.ufl_element()
        lagrange_space = ufl_element.family() == 'Lagrange'
        first_order = ufl_element.degree() == 1
        one_dim_mesh = self.physical_dim == 1

        # Raise error if the conditions above are not satisfied
        if not (lagrange_space and first_order and one_dim_mesh):
            raise NotImplementedError(
                "Envelope plot is only implemented for Lagrange finite element "+"space of order 1 and 1D meshes")
        
        # degrees of freedom (dofs) to vertex map
        d2v = dl.dof_to_vertex_map(self.function_space)

        return Continuous1D(self.mesh.coordinates().reshape(-1))._plot_envelope(lo_values[d2v], up_values[d2v], **kwargs)


class FEniCSMappedGeometry(MappedGeometry):
    """
    """
    @property
    def function_space(self):
        return self.geometry.function_space
    
    @property
    def funvec_shape(self):
        """The shape of the geometry (shape of the vector representation of the
        function value)."""
        return self.geometry.funvec_shape
    
    def par2fun(self,p):
        funvals = self.geometry.par2fun(p)
        if isinstance(funvals, dl.function.function.Function):
            funvals = [funvals]
        mapped_value_list = []
        for idx in range(len(funvals)):
            mapped_value = self.map(funvals[idx]) 
            if isinstance(mapped_value, ufl.algebra.Operator):
                mapped_value_list.append(dl.project(mapped_value, self.geometry.function_space))
            elif isinstance(mapped_value,dl.function.function.Function):
                mapped_value_list.append(mapped_value)
            else:
                raise ValueError(f"'{self.__class__.__name__}.map' should return 'ufl.algebra.Operator'")
            
        if len(mapped_value_list) == 1:
            return mapped_value_list[0]
        else:
            return mapped_value_list
    
    def fun2par(self,f):
        raise NotImplementedError
    
    def fun2vec(self,f):
        return self.geometry.fun2vec(f)

    def vec2fun(self,funvec):
        return self.geometry.vec2fun(funvec)


class MaternKLExpansion(_WrappedGeometry):
    """A geometry class that builds spectral representation of Matern covariance operator on the given input geometry. We create the representation using the stochastic partial differential operator, equation (15) in (Roininen, Huttunen and Lasanen, 2014). Zero Neumann boundary conditions are assumed for the stochastic partial differential equation (SPDE) and the default value of the smoothness parameter :math:`\\nu` is set to 0.5. To generate Matern field realizations, the method :meth:`par2field` is used. The input `p` of this method need to be an `n=dim` i.i.d random variables that follow a normal distribution. 

    For more details about the formulation of the SPDE see: Roininen, L., Huttunen, J. M., & Lasanen, S. (2014). Whittle-Matérn priors for Bayesian statistical inversion with applications in electrical impedance tomography. Inverse Problems & Imaging, 8(2), 561.

    Parameters
    -----------
    geometry : cuqi.fenics.geometry.Geometry
        An input geometry on which the Matern field representation is built (the geometry must have a mesh attribute)

    length_scale : float
        Length scale parameter (controls correlation length)

    num_terms: int
        Number of expansion terms to represent the Matern field realization

    nu : float, default 0.5 
        Smoothness parameter of the Matern field, must be greater then
        zero.

    boundary_conditions : str
        Boundary conditions for the SPDE. Currently 'Neumann' for zero flux, and 'zero' for zero Dirichlet boundary conditions are supported.

    normalize : bool, default True
        If True, the Matern field expansion modes are normalized to have a unit norm.

    Example
    -------
    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from cuqi.fenics.geometry import MaternKLExpansion, FEniCSContinuous
        from cuqi.distribution import Gaussian
        import dolfin as dl
        
        mesh = dl.UnitSquareMesh(20,20)
        V = dl.FunctionSpace(mesh, 'CG', 1)
        geometry = FEniCSContinuous(V)
        MaternGeometry = MaternKLExpansion(geometry, 
                                        length_scale = .2,
                                        num_terms=128)
        
        MaternField = Gaussian(mean=np.zeros(MaternGeometry.dim),
                        cov=np.eye(MaternGeometry.dim),
                        geometry= MaternGeometry)
        
        samples = MaternField.sample()
        samples.plot()

    """

    def __init__(self, geometry, length_scale, num_terms, nu=0.5, boundary_conditions='Neumann', normalize=True): 

        if nu <= 0:
            raise ValueError("Smoothness parameter nu must be positive")
        
        if not isinstance(geometry, (FEniCSMappedGeometry, FEniCSContinuous)):
            raise ValueError("Matern KL expansion is only implemented "+ 
                             "for cuqipy_fenics geometries")
        super().__init__(geometry)
        if not hasattr(geometry, 'mesh'):
            raise NotImplementedError
        self._length_scale = length_scale
        self._nu = nu
        self._num_terms = num_terms
        self._eig_val = None
        self._eig_vec = None
        self._boundary_conditions = boundary_conditions
        self._normalize = normalize

    @property
    def funvec_shape(self):
        """The shape of the geometry (shape of the vector representation of the
        function value)."""
        return self.geometry.funvec_shape
    
    @property
    def par_shape(self):
        return (self.num_terms,)

    @property
    def length_scale(self):
        return self._length_scale

    @property
    def nu(self):
        return self._nu

    @property
    def num_terms(self):
        return self._num_terms

    @property
    def function_space(self):
        return self.geometry.function_space
    
    @property
    def eig_val(self):
        return self._eig_val

    @property
    def eig_vec(self):
        return self._eig_vec

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @property
    def normalize(self):
        return self._normalize
    
    @property
    def physical_dim(self):
        """Returns the physical dimension of the geometry, e.g. 1, 2 or 3"""
        return self.geometry.physical_dim

    def __repr__(self) -> str:
        return "{} on {}".format(self.__class__.__name__,self.geometry.__repr__())

    def par2fun(self,p):
        return self.geometry.par2fun(self.par2field(p))

    def fun2vec(self,fun):
        """ Maps the function value (FEniCS object) to the corresponding vector
        representation of the function (ndarray of the function DOF values)."""
        return self.geometry.fun2vec(fun)
    
    def vec2fun(self,funvec):
        """ Maps the vector representation of the function (ndarray of the
        function DOF values) to the function value (FEniCS object)."""
        return self.geometry.vec2fun(funvec)

    def gradient(self, direction, wrt):
        direction = self.geometry.gradient(direction, wrt)
        return np.diag( np.sqrt(self.eig_val)).T@self.eig_vec.T@direction
        
    def par2field(self, p):
        """Applies linear transformation of the parameters p to
        generate a realization of the Matern field (given that p is a
        sample of `n=dim` i.i.d random variables that follow a normal
        distribution)"""

        if self._eig_vec is None and self._eig_val is None:
            self._build_basis() 
	   
        p = self._process_values(p)
        Ns = p.shape[-1]
        field_list = np.empty((self.geometry.par_dim,Ns))

        nu = self.nu
        d = self.physical_dim
        for idx in range(Ns):
            # For more details about the formulation below, see section 4.3 in
            # Chen, V., Dunlop, M. M., Papaspiliopoulos, O., & Stuart, A. M.
            # (2018). Dimension-robust MCMC in Bayesian inverse problems.
            # arXiv preprint arXiv:1803.03344.
            field_list[:,idx] = self.eig_vec@( self.eig_val**((nu+d/2)/2)*p[...,idx] )

        if len(field_list) == 1:
            return field_list[0]
        else:
            return field_list

    def _build_basis(self):
        """Builds the basis of expansion of the Matern covariance operator"""
        # Define function space, test and trial functions
        V = self.function_space
        u = dl.TrialFunction(V)
        v = dl.TestFunction(V)

        # Define the weak form a of the differential operator used in building the Matern field basis
        tau2 = 1/self.length_scale/self.length_scale
        a = tau2*u*v*dl.dx + dl.inner(dl.grad(u), dl.grad(v))*dl.dx
        
        # Set up the boundary conditions of the SPDE
        if self.boundary_conditions.lower() == 'neumann':
            boundary = lambda x, on_boundary: False
        elif self.boundary_conditions.lower() == 'zero':
            boundary = lambda x, on_boundary: on_boundary
        else:
            raise ValueError(f"Boundary conditions {self.boundary_conditions}, is not supported")

        u0 = dl.Constant('0.0')
        bc = dl.DirichletBC(V, u0, boundary)
        
        # Assemble the differential operator
        u_fun = dl.Function(V)
        L = u_fun*v*dl.dx
        K = dl.PETScMatrix()
        dl.assemble_system(a, L, bc, A_tensor=K)

        # Compute the first self.num_terms eigenvalues and eigenvectors of the 
        # (inverse) of the differential operator
        eigen_solver = dl.SLEPcEigenSolver(K)
        eigen_solver.parameters['spectrum'] = 'smallest magnitude'

        eigen_solver.solve(self.num_terms)
        self._eig_val = np.zeros(self.num_terms)
        self._eig_vec = np.zeros( [ u_fun.vector().get_local().shape[0], self.num_terms ] )

        for i in range( self.num_terms ):
            val, c, vec, cx = eigen_solver.get_eigenpair(i)
            self._eig_val[i] = val
            self._eig_vec[:,i] = vec.get_local()

        self._eig_val = np.reciprocal( self._eig_val )

        # Normalize the eigenvectors if required
        if self.normalize:
            self._eig_vec /= np.linalg.norm( self._eig_vec )
