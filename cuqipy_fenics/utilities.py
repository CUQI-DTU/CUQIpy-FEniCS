import dolfin as dl
from numbers import Number

class ExpressionFromCallable(dl.UserExpression):
    """A `dolfin.Expression` that is created from a callable. The callable
    should take a single argument, the coordinate vector `x` and return a
    number, the value of the expression at `x`. The keyword arguments are passed
    to the `dolfin.Expression` constructor. An example of the keyword arguments 
    is the `degree` argument for `dolfin.Expression` which is used to specify 
    the element (polynomial) degree of the expression.
    """
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def eval(self, value, x):
        value[:] = self.func(x)


def to_dolfin_expression(value, **expression_kwargs):
    """Converts a value to a `dolfin.Expression`. The value should be a 
    callable, `dolfin.Expression` or a number. expression_kwargs are passed to
    the `dolfin.Expression` constructor if the value is a callable. An example 
    of the keyword arguments is the `degree` argument for `dolfin.Expression` 
    which is used to specify the element (polynomial) degree of the expression.
    """
    
    if not callable(value) and expression_kwargs != {}:
        raise ValueError("Cannot pass kwargs to non-callable value")
    if callable(value):
        return ExpressionFromCallable(value, **expression_kwargs)
    elif isinstance(value, dl.UserExpression) \
        or isinstance(value, dl.Expression)\
            or isinstance(value, dl.Constant):
        return value
    elif isinstance(value, Number):
        return dl.Constant(value)
    else:
        raise ValueError("Cannot convert to dolfin.Expression")

