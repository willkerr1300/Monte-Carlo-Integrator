#include <pybind11/pybind11.h>
#include <random>
#include <omp.h>
#include <vector>

namespace py = pybind11;

// Example integrand: sum of coordinates
double example_function(const std::vector<double>& x) {
    double s = 0;
    for(auto xi : x) s += xi;
    return s;
}

// Monte Carlo batch integrator
double monte_carlo_batch(int dim, long n_samples, unsigned int seed) {
    double sum = 0.0;

    #pragma omp parallel
    {
        std::mt19937 rng(seed + omp_get_thread_num());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double thread_sum = 0.0;

        #pragma omp for
        for(long i = 0; i < n_samples; ++i){
            std::vector<double> point(dim);
            for(int j = 0; j < dim; ++j)
                point[j] = dist(rng);
            thread_sum += example_function(point);
        }

        #pragma omp atomic
        sum += thread_sum;
    }

    return sum;
}

PYBIND11_MODULE(monte_carlo_batch, m) {
    m.def("batch_integrate", &monte_carlo_batch, "Monte Carlo Batch Integrator",
          py::arg("dim"), py::arg("n_samples"), py::arg("seed") = 42);
}
