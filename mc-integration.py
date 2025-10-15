import numpy as np

def integrate(f, bounds, N=10_000_000):
  #determine number of dimensions in problem
  dims = len(bounds)
  #generate N n-dimensional random points 
  samples = np.random.rand(num_samples, dim)

  for i in range(dim):
    low, high = bounds[i]
    samples[:, i] = low + (high - low) * samples[:, i]

  # Evaluate the function at all points
  values = np.apply_along_axis(func, 1, samples)

  # Volume of the integration region
  volume = np.prod([high - low for (low, high) in bounds])

  # Monte Carlo estimate
  integral = volume * np.mean(values)
  return integral


def test_func(x):
    return x[0] * x[1] * x[2]

bounds = [(0, 1), (0, 1), (0, 1)]
estimate = monte_carlo_integrate(test_func, bounds, num_samples=1000000)
print("Estimated integral:", estimate)
