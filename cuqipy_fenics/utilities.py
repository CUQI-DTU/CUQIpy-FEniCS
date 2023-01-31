import dolfin as dl

class ExpressionFromCallable(dl.UserExpression):
    def __init__(self, callable, **kwargs):
        super().__init__(**kwargs)
        self.callable = callable

    def eval(self, value, x):
        value[:] = self.callable(x)

