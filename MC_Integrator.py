import numpy as np
import monte_carlo_batch

def monte_carlo_integrate(dim, n_samples, batch_size=10**6, seed=42):
    total_sum = 0.0
    n_batches = (n_samples + batch_size - 1) // batch_size

    for i in range(n_batches):
        current_batch = min(batch_size, n_samples - i * batch_size)
        # Call C++ batch integrator
        batch_sum = monte_carlo_batch.batch_integrate(dim, current_batch, seed + i)
        total_sum += batch_sum

    return total_sum / n_samples
