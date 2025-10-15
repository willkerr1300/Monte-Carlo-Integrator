from MC_Integrator import monte_carlo_integrate

dim = 5
n_samples = 10_000_000

result = monte_carlo_integrate(dim, n_samples, batch_size=1_000_000)

print("Monte Carlo estimate:", result)
