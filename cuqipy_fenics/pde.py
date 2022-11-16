import numpy as np
from abc import ABC, abstractmethod
from cuqi.pde import PDE
from cuqi.samples import CUQIarray
import dolfin as dl
import ufl
from copy import copy

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
    PDE_form : python function handle
        Function handle of a python function that returns the FEniCS weak form of the PDE.

        This python function takes as input, in this given order, the Bayesian parameter (input of the forward model), the state variable (solution variable), the adjoint variable (which is also the test variable in the weak formulation) as FEniCS functions (or FEniCS trail or test functions). See, for example, `demos/demo03_poisson_circular.py` for an example of how to define this function.

    mesh : FEniCS mesh
        FEniCS mesh object that defines the discretization of the domain.

    solution_function_space : FEniCS function space
        FEniCS function space object that defines the function space of the state variable (solution variable).

    parameter_function_space : FEniCS function space
        FEniCS function space object that defines the function space of the Bayesian parameter (input of the forward model).

    dirichlet_bc : FEniCS Dirichlet boundary condition object
        FEniCS Dirichlet boundary condition object that defines the Dirichlet boundary conditions of the PDE.

    adjoint_dirichlet_bc : FEniCS Dirichlet boundary condition object, required if the gradient is to be computed
        FEniCS Dirichlet boundary condition object that defines the Dirichlet boundary conditions of the adjoint PDE.

    observation_operator : python function handle, optional
        Function handle of a python function that returns the observed quantity from the PDE solution. If not provided, the identity operator is assumed (i.e. the entire solution is observed).
        This python function takes as input the Bayesian parameter (input of the forward model) and the state variable (solution variable) as first and second inputs, respectively.

        The returned observed quantity can be a ufl.algebra.Operator, FEniCS Function, np.ndarray, int, or float.

        Examples of `observation_operator` are `lambda m, u: m*u` and `lambda m, u: m*dl.sqrt(dl.inner(dl.grad(u),dl.grad(u)))`

    Example
    --------
    See `demos/demo03_poisson_circular.py` for an example of how to define a `cuqipy_fenics.pde` objects.

    """

    def __init__(self, PDE_form, mesh, solution_function_space, parameter_function_space, dirichlet_bc, adjoint_dirichlet_bc=None, observation_operator=None, lhs_form=None, rhs_form=None, reuse_assembled=False, linalg_solve=None, linalg_solve_kwargs=None):

        if (PDE_form is not None) == (lhs_form is not None and rhs_form is not None):
            raise ValueError('Either PDE_form or lhs_form and rhs_form should be provided, but not both.')

        # Construct the PDE weak form for printing
        if PDE_form is None:
            PDE_form = lambda m, u, p: lhs_form(m, u, p) - rhs_form(m, u, p)

        self.PDE_form = PDE_form # function of PDE_solution, PDE_parameter, test_function
        self.lhs_form = lhs_form
        self.rhs_form = rhs_form

        self.reuse_assembled = reuse_assembled

        self.mesh = mesh 
        self.solution_function_space  = solution_function_space
        self.parameter_function_space = parameter_function_space
        self.dirichlet_bc  = dirichlet_bc
        self.adjoint_dirichlet_bc = adjoint_dirichlet_bc
        self.observation_operator = self._create_observation_operator(observation_operator)

        #TODO: apply kwargs
        if linalg_solve is None:
            linalg_solve = dl.LUSolver()
        if linalg_solve_kwargs is None:
            linalg_solve_kwargs = {}
 
        self._solver = linalg_solve

        # Hack: use dolfin's solver label property to store status 
        # (whether the solver has correct operator)
        self._solver.rename("","invalid")
        self._linalg_solve_kwargs = linalg_solve_kwargs
        self.parameter = dl.Function(self.parameter_function_space)


    @property
    def parameter(self):
        """ Get the parameter of the PDE """
        return self._parameter
    
    @parameter.setter
    def parameter(self, value):
        """ Set the parameter of the PDE. Since the PDE solution depends on the parameter, this will set the PDE solution to None. """
        if value is None:
            raise ValueError('Parameter cannot be None.')
        
        # First time setting the parameter
        if not hasattr(self, '_parameter'):
            self._parameter = value

        # Subsequent times setting the parameter (avoid assigning the parameter to new object, set parameter array in place instead)
        elif self._is_parameter_updated(value):
            self._parameter.vector().set_local(value.vector().get_local())
            # The operator in the solver is no longer valid
            self._solver.rename("","invalid")

        # Clear the solution and the rhs
        self._forward_solution = None
        self._adjoint_solution = None
        self._gradient = None
        self.rhs = None


    @property
    def forward_solution(self):
        """ Get the forward solution of the PDE """
        return self._forward_solution

    @forward_solution.setter
    def forward_solution(self, value):
        """ Set the forward solution of the PDE """
        self._forward_solution = value

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

    def _is_parameter_updated(self, value):
        """ A helper function to check if the PDE model parameter (the parameter to be inferred) is updated """

        if hasattr(self, '_parameter') \
            and np.allclose(self._parameter.vector().get_local(),
             value.vector().get_local(), atol=1e-16, rtol=1e-16):
            return False
        else:
            return True

class SteadyStateLinearFEniCSPDE(FEniCSPDE):
    """ Class representation of steady state linear PDEs defined in FEniCS. It accepts the same arguments as the base class `cuqipy_fenics.pde.FEniCSPDE`."""

    def assemble(self, parameter=None):
        self._solution_trial_function = dl.TrialFunction(self.solution_function_space)
        self._solution_test_function = dl.TestFunction(self.solution_function_space)

        if parameter is not None:
            self.parameter = parameter

        if self.lhs_form is not None:
            self._assemble_lhs()
            self._assemble_rhs()

        else:
            self._assemble_full()



    def with_updated_rhs(self, rhs_form):
        """ """
        new_pde = copy(self)
        new_pde.rhs_form = rhs_form
        new_pde.rhs = None
        return new_pde


    def _assemble_full(self):
        if self.reuse_assembled\
                and self._solver.label() == "valid" and\
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
        
        self.dirichlet_bc.apply(diff_op)
        self.dirichlet_bc.apply(self.rhs)
        self._solver.set_operator(diff_op)
        self._solver.rename("","valid")


    def _assemble_lhs(self):
        if self.reuse_assembled\
                and self._solver.label() == "valid":
            return

        diff_op = dl.assemble(self.lhs_form(self.parameter,
                                            self._solution_trial_function,
                                            self._solution_test_function))

        self.dirichlet_bc.apply(diff_op)
        self._solver.set_operator(diff_op)
        self._solver.rename("","valid")


    def _assemble_rhs(self):
        if self.reuse_assembled\
                and self.rhs is not None:
            return

        self.rhs = dl.assemble(self.rhs_form(self.parameter,
                                            self._solution_test_function))
        self.dirichlet_bc.apply(self.rhs)


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
        if self.adjoint_dirichlet_bc is None:
            raise ValueError(
                "The adjoint Dirichlet boundary conditions are not defined.")

        # Create needed functions
        trial_adjoint = dl.TrialFunction(self.solution_function_space)
        adjoint = dl.Function(self.solution_function_space)

        # Compute forward solution
        # TODO: Use stored forward solution if available and wrt == self.parameter
        self.parameter = wrt
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
            self.adjoint_dirichlet_bc,
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
