import numpy as np
from abc import ABC, abstractmethod
from cuqi.pde import PDE
from cuqi.array import CUQIarray
import dolfin as dl
from copy import copy
import warnings
from .utilities import _import_ufl
ufl = _import_ufl()

__all__ = [
    'FEniCSPDE',
    'SteadyStateLinearFEniCSPDE'
]

class FEniCSPDE(PDE,ABC):
    """ Base class that represents PDEs problem defined in FEniCS. This class is not meant to be used directly,
    but rather it defines the API that should be implemented by subclasses for specific types of PDEs.

    This class along with the `cuqipy_fenics.geometry`, interfaces FEniCS PDE models with CUQIpy library.

    Parameters
    ----------
    PDE_form : callable or tuple of two callables
        If passed as a callable: the callable returns the weak form of the PDE.
        The callable should take three arguments, the first argument is the 
        parameter (input of the forward model), the second argument is the state
        variable (solution variable), and the third argument is the adjoint
        variable (the test variable in the weak formulation).

        If passed as a tuple of two callables: the first callable returns the 
        weak form of the PDE left hand side, and the second callable returns the
        weak form of the PDE right hand side. The left hand side callable takes
        the same three arguments as described above. The right hand side 
        callable takes only the parameter and the adjoint variable as arguments.
        See the example below.

    mesh : FEniCS mesh
        FEniCS mesh object that defines the discretization of the domain.

    solution_function_space : FEniCS function space
        FEniCS function space object that defines the function space of the state variable (solution variable).

    parameter_function_space : FEniCS function space
        FEniCS function space object that defines the function space of the Bayesian parameter (input of the forward model).

    dirichlet_bcs: FEniCS Dirichlet boundary condition object or a list of them
        FEniCS Dirichlet boundary condition object(s) that define the Dirichlet boundary conditions of the PDE.

    adjoint_dirichlet_bcs : FEniCS Dirichlet boundary condition object or a list of them, required if the gradient is to be computed
        FEniCS Dirichlet boundary condition object(s) that define the Dirichlet boundary conditions of the adjoint PDE.

    observation_operator : python function handle, optional
        Function handle of a python function that returns the observed quantity from the PDE solution. If not provided, the identity operator is assumed (i.e. the entire solution is observed).
        This python function takes as input the Bayesian parameter (input of the forward model) and the state variable (solution variable) as first and second inputs, respectively.

        The returned observed quantity can be a ufl.algebra.Operator, FEniCS Function, np.ndarray, int, or float.

        Examples of `observation_operator` are `lambda m, u: m*u` and `lambda m, u: m*dl.sqrt(dl.inner(dl.grad(u),dl.grad(u)))`

    reuse_assembled : bool, optional
        Flag to indicate whether the assembled (and possibly factored) 
        differential operator should be reused when the parameter is not 
        changed.
        If True, the assembled matrices are reused. If False, the assembled 
        matrices are not reused. Default is False.

    linalg_solve : FEniCS linear solver object, optional

    linalg_solve_kwargs : dict, optional
        Dictionary of keyword arguments to be passed to the linear solver object.

    Example
    --------

    .. code-block:: python

        # Define mesh
        mesh = dl.UnitSquareMesh(20, 20)
        
        # Set up function spaces
        solution_function_space = dl.FunctionSpace(mesh, 'Lagrange', 2)
        parameter_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1)
        
        # Set up Dirichlet boundaries
        def u_boundary(x, on_boundary):
            return on_boundary
        
        dirichlet_bc_expr = dl.Expression("0", degree=1) 
        dirichlet_bcs = dl.DirichletBC(solution_function_space,
                                      dirichlet_bc_expr,
                                      u_boundary)
        
        # Set up PDE variational form
        def lhs_form(m,u,p):
            return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx 
        
        def rhs_form(m,p):
            return - dl.Constant(1)*p*ufl.dx
        
        # Create the PDE object 
        PDE = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE( 
                (lhs_form, rhs_form),
                mesh, 
                parameter_function_space=parameter_function_space,
                solution_function_space=solution_function_space,
                dirichlet_bcs=dirichlet_bcs)
            

    """

    def __init__(self, PDE_form, mesh, solution_function_space,
                 parameter_function_space, dirichlet_bcs, adjoint_dirichlet_bcs=None,
                 observation_operator=None, reuse_assembled=False, linalg_solve=None,
                 linalg_solve_kwargs=None):

        if PDE_form is None:
            raise ValueError('PDE_form should be provided and cannot be None.')

        self._form = PDE_form

        self.mesh = mesh
        self.solution_function_space = solution_function_space
        self.parameter_function_space = parameter_function_space
        self._dirichlet_bcs = dirichlet_bcs if isinstance(
            dirichlet_bcs, list) else [dirichlet_bcs]
        self._adjoint_dirichlet_bcs = adjoint_dirichlet_bcs if isinstance(
            adjoint_dirichlet_bcs, list) else [adjoint_dirichlet_bcs]
        self.observation_operator = observation_operator
        self.reuse_assembled = reuse_assembled

        if linalg_solve is None:
            linalg_solve = dl.LUSolver()
        if linalg_solve_kwargs is None:
            linalg_solve_kwargs = {}

        self._solver = linalg_solve

        # Flag to store whether the solver has correct operator
        # initially is set to False. These flags are shared between
        # shallow copies of this class.
        self._flags = {"is_operator_valid": False}

        # Set the solver parameters
        self._linalg_solve_kwargs = linalg_solve_kwargs
        for key, value in linalg_solve_kwargs.items():
            self._solver.parameters[key] = value

        # Initialize the parameter
        self.parameter = dl.Function(self.parameter_function_space)


    @property
    def parameter(self):
        """ Get the parameter of the PDE """
        return self._parameter
    
    @parameter.setter
    def parameter(self, value):
        """ Set the parameter of the PDE. Since the PDE solution depends on the 
        parameter, this will set the PDE solution to None. """
        if value is None:
            raise ValueError('Parameter cannot be None.')

        # First time setting the parameter
        if not hasattr(self, '_parameter'):
            self._parameter = value

        # Subsequent times setting the parameter (avoid assigning the parameter
        # to new object, set parameter array in place instead)
        elif self._is_parameter_new(value):
            self._parameter.vector().set_local(value.vector().get_local())
            # The operator in the solver is no longer valid
            self._flags["is_operator_valid"] = False

        # Clear the solution and the rhs
        self._forward_solution = None
        self._adjoint_solution = None
        self._gradient = None
        self.rhs = None


    @property
    def forward_solution(self):
        """ Get the forward solution of the PDE """
        return self._forward_solution

    @property
    def PDE_form(self):
        """ Get the PDE form """
        if isinstance(self._form, tuple):
            return lambda m, u, p: self._form[0](m, u, p) - self._form[1](m, p)
        else:
            return self._form

    @property
    def lhs_form(self):
        """ Get the lhs form """
        if isinstance(self._form, tuple):
            return self._form[0]
        else:
            return None

    @property
    def rhs_form(self):
        """ Get the rhs form """
        if isinstance(self._form, tuple):
            return self._form[1]
        else:
            return None

    @property
    def reuse_assembled(self):
        """ Get the reuse_assembled flag """
        return self._reuse_assembled

    @reuse_assembled.setter
    def reuse_assembled(self, value):
        """ Set the reuse_assembled flag """
        if not isinstance(self._form, tuple) and value:
            raise ValueError('PDE_form should be a tuple of the lhs and rhs"+\
            "forms to be able to set the reuse_assembled flag to True.')
        self._reuse_assembled = value

    @rhs_form.setter
    def rhs_form(self, value):
        """ Set the rhs form """
        if isinstance(self._form, tuple):
            self._form = (self._form[0], value)
        else:
            raise ValueError('Cannot set rhs_form if PDE_form is not a tuple.')

    @forward_solution.setter
    def forward_solution(self, value):
        """ Set the forward solution of the PDE """
        self._forward_solution = value

    @property
    def observation_operator(self):
        """ Get the observation operator """
        return self._observation_operator

    @observation_operator.setter
    def observation_operator(self, value):
        """ Set the observation operator """
        self._observation_operator = self._create_observation_operator(value)

    @abstractmethod
    def assemble(self,parameter):
        """ Assemble the PDE weak form """
        raise NotImplementedError

    @abstractmethod
    def solve(self):
        """ Solve the PDE """
        raise NotImplementedError

    @abstractmethod
    def observe(self,PDE_solution):
        """ Apply observation operator on the PDE solution """
        raise NotImplementedError

    @abstractmethod
    def gradient_wrt_parameter(self):
        """ Compute gradient of the PDE weak form w.r.t. the parameter"""
        raise NotImplementedError 

    @abstractmethod
    def _create_observation_operator(self, observation_operator):
        raise NotImplementedError

    def _is_parameter_new(self, input_parameter):
        """ A helper function to check if the `input_parameter` is different 
        from the current parameter (cached in self._parameter). """

        if hasattr(self, '_parameter') \
            and np.allclose(self._parameter.vector().get_local(),
                            input_parameter.vector().get_local(),
                            atol=dl.DOLFIN_EPS, rtol=dl.DOLFIN_EPS):
            return False
        else:
            return True


class SteadyStateLinearFEniCSPDE(FEniCSPDE):
    """ Class representation of steady state linear PDEs defined in FEniCS. It accepts the same arguments as the base class `cuqipy_fenics.pde.FEniCSPDE`."""

    def assemble(self, parameter=None):
        self._solution_trial_function = dl.TrialFunction(
            self.solution_function_space)
        self._solution_test_function = dl.TestFunction(
            self.solution_function_space)

        if parameter is not None:
            self.parameter = parameter

        # Either assemble the lhs and rhs forms separately or the full PDE form
        if self.lhs_form is not None:
            self._assemble_lhs()
            self._assemble_rhs()

        else:
            self._assemble_full()



    def with_updated_rhs(self, rhs_form):
        """ A method to create a shallow copy of the PDE model with updated rhs 
        form. The user can set the flag `reuse_assembled` to True in both PDE
        objects to allow the two PDE objects to reuse the factorized 
        differential operators. """

        # Warn the user if reuse_assembled is set to False
        if not self.reuse_assembled:
            warnings.warn('The flag `reuse_assembled` is set to False. '+\
                'The new PDE object will not be able to reuse the '+\
                'factorized differential operators from the current PDE'+\
                'object.')

        new_pde = copy(self)
        new_pde.rhs_form = rhs_form
        new_pde.rhs = None
        return new_pde

    def _assemble_full(self):
        """ Assemble the full PDE form """
        if self.reuse_assembled\
                and self._flags["is_operator_valid"] and\
                self.rhs is not None:
            return

        diff_op = dl.lhs(self.PDE_form(self.parameter,
                                       self._solution_trial_function,
                                       self._solution_test_function))
        self.rhs = dl.rhs(self.PDE_form(self.parameter,
                                        self._solution_trial_function,
                                        self._solution_test_function))

        if self.rhs.empty():
            self.rhs = dl.Constant(0)*self._solution_test_function*dl.dx

        diff_op = dl.assemble(diff_op)
        self.rhs = dl.assemble(self.rhs)
        for bc in self._dirichlet_bcs: 
            bc.apply(diff_op)
            bc.apply(self.rhs)
        self._solver.set_operator(diff_op)
        self._flags["is_operator_valid"] = True

    def _assemble_lhs(self):
        """ Assemble the lhs form """
        if self.reuse_assembled\
                and self._flags["is_operator_valid"]:
            return

        diff_op = dl.assemble(self.lhs_form(self.parameter,
                                            self._solution_trial_function,
                                            self._solution_test_function))

        for bc in self._dirichlet_bcs: bc.apply(diff_op)
        self._solver.set_operator(diff_op)
        self._flags["is_operator_valid"] = True

    def _assemble_rhs(self):
        """ Assemble the rhs form """
        if self.reuse_assembled\
                and self.rhs is not None:
            return

        self.rhs = dl.assemble(self.rhs_form(self.parameter,
                                             self._solution_test_function))
        for bc in self._dirichlet_bcs: bc.apply(self.rhs)


    def solve(self):
        self.forward_solution = dl.Function(self.solution_function_space)       
        self._solver.solve(self.forward_solution.vector(), self.rhs)
        return self.forward_solution, None

    def observe(self,PDE_solution_fun):
        if self.observation_operator is None: 
            return PDE_solution_fun
        else:
            return self._apply_obs_op(self.parameter, PDE_solution_fun)

    def gradient_wrt_parameter(self, direction, wrt, **kwargs):
        """ Compute the gradient of the PDE with respect to the parameter

        Note: This implementation is largely based on the code:
        https://github.com/hippylib/hippylib/blob/master/hippylib/modeling/PDEProblem.py

        See also: Gunzburger, M. D. (2002). Perspectives in flow control and optimization. Society for Industrial and Applied Mathematics, for adjoint based derivative derivation. 
        """
        # Raise an error if the adjoint boundary conditions are not provided
        if self._adjoint_dirichlet_bcs is None:
            raise ValueError(
                "The adjoint Dirichlet boundary conditions are not defined.")

        # Create needed functions
        trial_adjoint = dl.TrialFunction(self.solution_function_space)
        adjoint = dl.Function(self.solution_function_space)

        # Compute forward solution
        # TODO: Use stored forward solution if available and wrt == self.parameter
        self.parameter = wrt
        self.assemble()
        self.forward_solution, _ = self.solve()

        # Compute adjoint solution
        test_parameter = dl.TestFunction(self.parameter_function_space)
        test_solution = dl.TestFunction(self.solution_function_space)

        # note: temp_form is a weak form used for building the adjoint operator
        temp_form = self.PDE_form(wrt, self.forward_solution, trial_adjoint)
        adjoint_form = dl.derivative(
            temp_form, self.forward_solution, test_solution)

        adjoint_matrix, _ = dl.assemble_system(
            adjoint_form,
            ufl.inner(self.forward_solution, test_solution) * ufl.dx,
            self._adjoint_dirichlet_bcs,
        )

        #TODO: account for observation operator
        if self.observation_operator is not None:
            raise NotImplementedError(
                "Gradient wrt parameter for PDE with observation operator not implemented")

        adjoint_rhs = -direction.vector()
        dl.solve(adjoint_matrix, adjoint.vector(), adjoint_rhs)

        # Compute gradient
        # note: temp_form is a weak form used for building the gradient
        temp_form = self.PDE_form(wrt, self.forward_solution, adjoint)
        gradient_form = dl.derivative(temp_form, wrt, test_parameter)
        gradient = dl.Function(self.parameter_function_space)
        dl.assemble(gradient_form, tensor=gradient.vector())
        return gradient


    def _apply_obs_op(self, PDE_parameter_fun, PDE_solution_fun,):
        obs = self.observation_operator(PDE_parameter_fun, PDE_solution_fun)
        if isinstance(obs, ufl.algebra.Operator):
            return dl.project(obs, self.solution_function_space)
        elif isinstance(obs, dl.function.function.Function):
            return obs
        elif isinstance(obs, (np.ndarray, int, float)):
            return obs
        else:
            raise NotImplementedError("obs_op output must be a number, a numpy array or a ufl.algebra.Operator type")
    

    def _create_observation_operator(self, observation_operator):
        """
        """
        if observation_operator == 'potential':
            observation_operator = lambda m, u: u 
        elif observation_operator == 'gradu_squared':
            observation_operator = lambda m, u: dl.inner(dl.grad(u),dl.grad(u))
        elif observation_operator == 'power_density':
            observation_operator = lambda m, u: m*dl.inner(dl.grad(u),dl.grad(u))
        elif observation_operator == 'sigma_u':
            observation_operator = lambda m, u: m*u
        elif observation_operator == 'sigma_norm_gradu':
            observation_operator = lambda m, u: m*dl.sqrt(dl.inner(dl.grad(u),dl.grad(u)))
        elif observation_operator == None or callable(observation_operator):
            observation_operator = observation_operator
        else:
            raise NotImplementedError
        return observation_operator
