#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <math.h>
namespace mpi = boost::mpi;

int main()
{
  // Necessary for every MPI program
  mpi::environment env;
  mpi::communicator world;

  // A lambda expression to approximate pi
  auto approx = [](double a){ return 4.0 / (1.0 + a * a); };

  // Some constants and other variables
  const double exact_pi{4.0 * atan(1.0)};
  const int num_grid{1000000000};
  double approx_pi{0}, to_reduce_pi{0}, d_grid{0}, sum{0}, inner_temp{0};

  // Start the timer
  mpi::timer timer;

  // Approximate pi over the grid
  d_grid = 1.0 / (double) num_grid;
  for (int i = world.rank(); i < num_grid; i += world.size()){
    inner_temp = d_grid * ((double) i + 0.5);
    sum += approx(inner_temp);
  }
  to_reduce_pi = d_grid * sum;
  
  // Reduction of to_reduce_pi to approx_pi
  all_reduce(world, to_reduce_pi, approx_pi, std::plus<double>());

  // Print the approx_pi and elapsed time
  if (world.rank() == 0) {
    std::cout << boost::format("Pi is approximately %.16f, error is %.16f\n")
      % approx_pi % (exact_pi - approx_pi);
    std::cout << boost::format("-> Elapsed time %.16fs\n") % timer.elapsed();
  }

  return 0;
}
