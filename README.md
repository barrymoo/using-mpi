Purpose
---

Working through "Using MPI" 3rd edition with Boost.MPI. I would like to learn
some of the c++11 and c++14 standard in the process. We'll see how it goes :P

Projects:

1. `midpoint_approx_pi` (3.1): Use the midpoint rule to approximate Pi. On Arch Linux
   (gcc 6.2.1, openmpi 1.10.4, and boost/boost-libs 1.62.0):
```
cmake -DCMAKE_CXX_COMPILER=mpicxx ..
make
mpirun -np <num_cpus> midpoint_approx_pi
```

2. `self_schedule_matrix_vector_multiply` (3.6): Calculate $c = A \times b$,
    matrix and vectors are stored as vectors because boost::multi_array has
    no serialization definitions (something I need to learn about apparently).
    I am at the testing operations with vector<vector <double>> in Boost::MPI.
