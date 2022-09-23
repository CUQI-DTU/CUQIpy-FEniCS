import numpy as np
from abc import ABC, abstractmethod
from cuqi.pde import PDE
from cuqi.samples import CUQIarray
import dolfin as dl
import ufl

__all__ = [
    'FEniCSPDE',
    'SteadyStateLinearFEniCSPDE'
]

class FEniCSPDE(PDE,ABC):
    def __init__(self, PDE_form, mesh, solution_function_space, parameter_function_space, dirichlet_bc, adjoint_dirichlet_bc=None, observation_operator=None):
        self.PDE_form = PDE_form # function of PDE_solution, PDE_parameter, test_function
        self.mesh = mesh 
        self.solution_function_space  = solution_function_space
        self.parameter_function_space = parameter_function_space
        self.dirichlet_bc  = dirichlet_bc
        self.adjoint_dirichlet_bc = adjoint_dirichlet_bc
        self.observation_operator = self._create_observation_operator(observation_operator)

    @property
    def parameter(self):
        """ Get the parameter of the PDE """
        return self._parameter
    
    @parameter.setter
    def parameter(self, value):
        """ Set the parameter of the PDE """
        self._parameter = value
        self._forward_solution = None

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

class SteadyStateLinearFEniCSPDE(FEniCSPDE):
    def __init__(self, PDE_form, mesh, solution_function_space, 
                 parameter_function_space, 
                 dirichlet_bc, adjoint_dirichlet_bc=None,
                observation_operator=None):
        super().__init__(PDE_form, mesh, solution_function_space,
                         parameter_function_space, 
                         dirichlet_bc, adjoint_dirichlet_bc,
                         observation_operator=observation_operator)


    def assemble(self, parameter=None):
        if parameter is not None:
            self.parameter = parameter

        solution_trial_function = dl.TrialFunction(self.solution_function_space)
        solution_test_function = dl.TestFunction(self.solution_function_space)
        self.diff_op, self.rhs  = \
            dl.lhs(self.PDE_form(self.parameter, solution_trial_function,solution_test_function)),\
            dl.rhs(self.PDE_form(self.parameter, solution_trial_function, solution_test_function))

    def solve(self):
        self.forward_solution = dl.Function(self.solution_function_space)
        dl.solve(self.diff_op ==  self.rhs, self.forward_solution, self.dirichlet_bc)
        return self.forward_solution, None

    def observe(self,PDE_solution_fun):
        if self.observation_operator is None: 
            return PDE_solution_fun
        else:
            return self._apply_obs_op(self.PDE_parameter_fun, PDE_solution_fun)

 
    def gradient_wrt_parameter(self, direction, wrt, **kwargs):
        """ Compute the gradient of the PDE with respect to the parameter

        Note: This implementation is largely based on the code:
        https://github.com/hippylib/hippylib/blob/master/hippylib/modeling/PDEProblem.py
        """
        # Create needed functions
        trial_adjoint = dl.TrialFunction(self.solution_function_space)
        adjoint = dl.Function(self.solution_function_space)

        # Compute forward solution
        # TODO: Use stored forward solution if available and wrt == self.parameter
        self.parameter = wrt 
        self.forward_solution,_ = self.solve() 

        # Compute adjoint solution
        test_parameter = dl.TestFunction(self.parameter_function_space)
        test_solution = dl.TestFunction(self.solution_function_space)
        
        J = self.PDE_form(wrt, self.forward_solution, trial_adjoint)
        dJdu = dl.derivative(
            J, self.forward_solution, test_solution)
        
        dJdu_matrix, _ = dl.assemble_system(dJdu,
                                      ufl.inner(
                                          self.forward_solution, test_solution)*ufl.dx, self.adjoint_dirichlet_bc)
        #TODO: account for observation operator
        if self.observation_operator is not None:
            raise NotImplementedError("Gradient wrt parameter for PDE with observation operator not implemented")
            
        adjoint_rhs = -direction.vector() 
        dl.solve(dJdu_matrix, adjoint.vector(), adjoint_rhs)

        # Compute gradient
        J2 = self.PDE_form(wrt, self.forward_solution, adjoint)
        gradient = dl.Function(self.parameter_function_space)
        dl.assemble(dl.derivative(J2, wrt, test_parameter), tensor=gradient.vector())
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
