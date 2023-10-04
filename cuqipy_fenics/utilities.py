import numpy as np
import dolfin as dl
import warnings
from numbers import Number
from cuqi.samples import Samples

def _import_ufl():
    """Import ufl module. This function is used to make importing ufl compatible
    with FEniCS versions that name `ufl` module as`ufl` or alternatively as 
    `ufl_legacy`.
    """
    try:
        import ufl_legacy as ufl
    except (Exception, RuntimeError, ImportError):
        import ufl
    return ufl

def _compute_stats(samples: Samples):
    """This function computes the statistics (mean and variance) of a set of 
    samples on the function value representation. Two different approaches 
    used in computing the variance: The first approach is computing the variance
    var1 using the FEniCS functions directly, and is general for any FEM 
    function. The second approach is computing the variance var2 using the
    FEniCS vectors directly, and is specific for some FEM function spaces,
    e.g. CG1. The returned values are the mean, var1, and var2.

    Note that the second approach gives the same result as what is
    obtained by the command `samples.funvals.vector.variance()`
    """

    geom = samples.geometry
    # raise error if geom does not have a function_space attribute
    if not hasattr(geom, 'function_space'):
        raise AttributeError("The geometry must have a "+\
                             "function_space attribute")
 
    V = geom.function_space

    # Loop to compute the samples function value
    Ns = samples.samples.shape[-1] # Number of samples
    sample_funs = []
    sample_funs_dof_vecs = np.empty((V.dim(), Ns)) 
    for i, sample in enumerate(samples):
        sample_funs.append(geom.par2fun(sample))
        sample_funs_dof_vecs[:, i] = sample_funs[-1].vector().get_local()

    # Sample mean
    sample_mean_dof = np.mean(sample_funs_dof_vecs, axis=1)
    sample_mean_f = dl.Function(V)
    sample_mean_f.vector().set_local(sample_mean_dof)

    # Compute variance 
    # Approach 1 (this approach is general for all FEM function spaces)

    # Create a function space with higher order elements
    V2 = dl.FunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree()*2)

    # Loop to create terms required for variance computation
    var_terms = np.empty((V2.dim(), Ns))
    for i, sample_f in enumerate(sample_funs):
        expr_f = dl.project(sample_f*sample_f - 2*sample_mean_f*sample_f, V2)
        var_terms[:, i] = expr_f.vector().get_local()
    
    mean_var_terms = np.mean(var_terms, axis=1)
    mean_var_terms_f = dl.Function(V2)
    mean_var_terms_f.vector().set_local(mean_var_terms)
    
    var1 = dl.project(mean_var_terms_f + sample_mean_f*sample_mean_f , V2)

    # Approach 2 (this approach is specific for some FEM function spaces, e.g. CG1)
    if V.ufl_element().family() != 'Lagrange':
        warnings.warn("The function space is not Lagrange, the variance, var2,"+ 
                     "computed using the second approach may not be correct.")
    var2_vec = np.var(sample_funs_dof_vecs, axis=1)
    var2 = dl.Function(V)
    var2.vector().set_local(var2_vec)

    return sample_mean_f, var1, var2


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
    # Note that dl.Expression and dl.Constant objects are
    # callable objects, so we need to check for them first
    if isinstance(value, dl.UserExpression) \
        or isinstance(value, dl.Expression)\
            or isinstance(value, dl.Constant):
        return value
    elif callable(value):
        return ExpressionFromCallable(value, **expression_kwargs)
    elif isinstance(value, Number):
        return dl.Constant(value)
    else:
        raise ValueError("Cannot convert to dolfin.Expression")
