
import cuqi 
import dolfin as dl

class ThetaMethod(cuqi.pde.TimeIntegrator):
    def __init__(self, theta):
	    # solver details
        self.theta = theta # 0: backward Euler, 1: forward Euler, 0.5: trapezoidal  

    def propagate(self, u, dt, rhs, diff_op, solver, I, apply_bc):
        u_next = dl.Function(u.function_space())
        u_next.vector().zero()

        rhs_op = I +self.theta*dt*diff_op
        rhs_op.mult(u.vector(), u_next.vector())

        u_next.vector().axpy(dt, rhs)

        lhs_op = I - (1-self.theta)*dt*diff_op
        apply_bc(lhs_op)
        apply_bc(u_next.vector())
        solver.set_operator(lhs_op)
        solver.solve(u_next.vector(), u_next.vector())
        return u_next, None

class ForwardEuler(ThetaMethod):
    def __init__(self):
        super().__init__(1.0)

class BackwardEuler(ThetaMethod):
    def __init__(self):
        super().__init__(0.0)

class Trapezoidal(ThetaMethod):
    def __init__(self):
        super().__init__(0.5)
