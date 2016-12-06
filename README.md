Purpose
---

Working through "Using MPI" 3rd edition with Boost.MPI. I would like to learn
some of the c++11 and c++14 standard in the process. We'll see how it goes :P

Projects:

1. `midpoint_approx_pi`: Use the midpoint rule to approximate Pi. On my Mac, I
   am using `Macports` with `openmpi` and `boost +openmpi`. My build:
```
cmake -DCMAKE_CXX_COMPILER=mpicxx-openmpi-mp ..
make
mpirun -np <num_cpus> midpoint_approx_pi
```
