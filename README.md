Purpose
---

Working through "Using MPI" 3rd edition with Boost.MPI. I would like to learn
some of the c++11 and c++14 standard in the process. We'll see how it goes :P

#### Compilers/Libraries Used

- GCC: v7.2.0
- OpenMPI: v2.1.1
- CMake: v3.9.6
- Boost/Boost Lib: v1.65.1

#### Compiling Projects

```
mkdir build; cd build
cmake ..
make
mpirun -np <num_cpus .gt. 1> ./<executable>
```

#### Projects

1. `midpoint_approx_pi` (3.1): Use the midpoint rule to approximate pi.

2. `self_schedule_matrix_vector_multiply` (3.6): Calculate `c = A \times b`
    - `with-vector.cpp`: Uses `std::vector` as the vector and matrix container
    - `with-multi-array.cpp`: Uses `boost::multi_array` as the vector matrix container

3. `self_schedule_matrix_matrix_multiply` (Personal Exercise): Calculate `C = A \times B`
    - `with-multi-array.cpp`: Uses `boost::multi_array` as the matrix container
    - `buffer` is now a 2 by rows matrix

#### Potential Enhancements

- Generalize the matrix-vector, matrix-matrix examples
    - Matrix-matrix can have rows and columns for both `A` and `B` (rows of `A` == cols of `B`)
    - Provide command line interface for input dimensions
- `<algorithm>` usage with `boost::multi_array` for reductions
- Better initialization for `boost::multi_array`
- Replace `boost::format` with [https://github.com/fmtlib/fmt](fmtlib) (for "funsies")
- Try out sanitizers (ASAN) to check for memory leaks, addressability, uninitialized memory
- Check for vectorization in worker code
- Check scalability
- Identify slow parts via profiling

#### Questions

- Is the serialization definition reasonable?
