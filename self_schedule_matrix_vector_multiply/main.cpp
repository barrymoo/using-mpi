#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <vector>

using namespace std;
namespace mpi = boost::mpi;

int main()
{
  // Necessary for every MPI program
  mpi::environment env;
  mpi::communicator world;

  // Vector and Matrix
  typedef vector<double> Vector;
  typedef vector<vector <double>> Matrix;

  // Variables and multi_array definitions
  const int rows{100}, cols{100};
  Vector a(cols), c(cols);
  Matrix B(rows, Vector(cols));

  // Start the timer
  mpi::timer timer;

  // Rank 0 is our manager, every one else is a worker.
  if (world.rank() == 0) {
    // Initialize a, B, and c to 0.0
    for (int i = 0; i < cols; ++i) {
      a[i] = i;
      c[i] = 1.0;
      for (int j = 0; j < rows; ++j) {
        B[j][i] = j;
      }
    }

    // Test send Matrix
    world.send(1, 0, B);
  }
  else {
    Matrix B(rows, Vector(cols));
    // Test receive Matrix
    world.recv(0, 0, B);
    cout << boost::format("B[4][4] is %.16f on rank %i\n") % B[4][4] % world.rank();
  }

  // Print the approx_pi and elapsed time
  if (world.rank() == 0) {
    cout << boost::format("-> Elapsed time %.16fs\n") % timer.elapsed();
  }

  return 0;
}
