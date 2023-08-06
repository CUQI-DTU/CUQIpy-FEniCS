import cuqipy_fenics

Ns = 100 # Number of samples
poisson_problem = cuqipy_fenics.testproblem.FEniCSPoisson2D((32, 32), 
                                    bc_types=['Dirichlet', 'Neumann', 'Dirichlet', 'Neumann'],
                                    bc_values=[0, 0, 0, 0],
                                    source_term=1.0,
                                    field_type='KL',
                                    mapping='exponential',
                                    field_params={'num_terms': 32, 'length_scale': 0.1})

# UQ method samples the posterior distribution of the poisson_problem
# and plots the credible interval, the mean and the variance of the solution.
# The sampling here might take a while (~13 min).
samples = poisson_problem.UQ(Ns, percent=97)
