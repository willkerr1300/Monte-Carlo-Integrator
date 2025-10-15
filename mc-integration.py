import numpy as np

def integrate(f, bounds, N=10_000_000):
  dim = len(bounds)
  samples = np.random.rand(N, dim)
  for i in range(dim):
    low, high = bounds[i]
    samples[:, i] = low + (high - low) * samples[:, i]

  values = np.apply_along_axis(f, 1, samples)

  volume = np.prod([high - low for (low, high) in bounds])

  integral = volume * np.mean(values)
  return integral


def integate2(f, bounds, N=10_000_000, batch_size=1_000_000):
    dim = len(bounds)
    total = 0.0
    n_batches = int(np.ceil(N / batch_size))
    volume = np.prod([high - low for (low, high) in bounds])
    for _ in range(n_batches):
        current_batch = min(batch_size, N)
        samples = np.random.rand(current_batch, dim)
        for i in range(dim):
            low, high = bounds[i]
            samples[:, i] = low + (high - low) * samples[:, i]
        total += np.sum(np.apply_along_axis(f, 1, samples))
        N -= current_batch
    
    integral = volume * total / (n_batches * batch_size)
    return integral

from concurrent.futures import ThreadPoolExecutor

def integrate3(f, bounds, N=10000000, batch_size=1000000, threads=4):
    dim = len(bounds)
    volume = np.prod([high - low for (low, high) in bounds])
    
    def compute_batch(n):
        samples = np.random.rand(n, dim)
        for i in range(dim):
            low, high = bounds[i]
            samples[:, i] = low + (high - low) * samples[:, i]
        return np.sum(np.apply_along_axis(f, 1, samples))
    
    batches = [min(batch_size, N - i*batch_size) for i in range(int(np.ceil(N / batch_size)))]
    
    total = 0.0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(compute_batch, batches)
        total = sum(results)
    
    integral = volume * total / N
    return integral

def test_func(x):
    return x[0] * x[1] * x[2]  # 3D integral over [0,1]^3

def main():
    bounds = [(0, 1), (0, 1), (0, 1)]
    N = 1_000_000
    batch_size = 100_000
    threads = 4

    print("Comparing Monte Carlo Integrators...\n")

    # Naive
    start = time.time()
    result1 = integrate(test_func, bounds, N)
    t1 = time.time() - start
    print(f"Naive:   Result={result1:.6f}, Time={t1:.4f}s")

    # Batched
    start = time.time()
    result2 = integrate2(test_func, bounds, N, batch_size)
    t2 = time.time() - start
    print(f"Batched: Result={result2:.6f}, Time={t2:.4f}s")

    # Parallel
    start = time.time()
    result3 = integrate3(test_func, bounds, N, batch_size, threads)
    t3 = time.time() - start
    print(f"Parallel: Result={result3:.6f}, Time={t3:.4f}s")

if __name__ == "__main__":
    main()
