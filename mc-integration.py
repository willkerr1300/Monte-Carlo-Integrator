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


def integate2(f, bounds, N=10_000_000, batches=1_000_000):
    dim = len(bounds)
    total = 0.0
    n_batches = int(np.ceil(num_samples / batch_size))
    volume = np.prod([high - low for (low, high) in bounds])
    
    for _ in range(n_batches):
        current_batch = min(batch_size, num_samples)
        samples = np.random.rand(current_batch, dim)
        for i in range(dim):
            low, high = bounds[i]
            samples[:, i] = low + (high - low) * samples[:, i]
        total += np.sum(np.apply_along_axis(func, 1, samples))
        num_samples -= current_batch
    
    integral = volume * total / (n_batches * batch_size)
    return integral
