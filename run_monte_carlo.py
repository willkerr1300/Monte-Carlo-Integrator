import time
from monte_carlo_wrapper import monte_carlo_integrate

dim = 5
n_samples = 10_000_000
batch_size = 1_000_000

start = time.perf_counter()
result = monte_carlo_integrate(dim, n_samples, batch_size)
end = time.perf_counter()

print(f"Monte Carlo estimate: {result}")
print(f"Time elapsed: {end - start:.3f} seconds")
