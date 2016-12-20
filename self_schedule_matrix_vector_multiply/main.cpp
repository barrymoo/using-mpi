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
  Matrix A(rows, Vector(cols));
  Vector b(cols), c(cols), buffer(cols);
  mpi::status s;
  int row;
  double result;

  // Start the timer
  mpi::timer timer;

  // Initialize on Rank 0 and broadcast column vector
  if (world.rank() == 0) {
    // Initialize a, B, and c to 0.0
    for (int i = 0; i < cols; ++i) {
      b[i] = i;
      c[i] = 0.0;
      for (int j = 0; j < rows; ++j) {
        A[i][j] = j;
      }
    }
  }
  mpi::broadcast(world, b, 0);


  // Rank 0 is our manager, every one else is a worker.
  if (world.rank() == 0) {
    // Our column counter
    int cols_sent = 0;

    // Send row to each worker process, cols_sent tells us what work is left
    for (int i = 0; i < min(world.size() - 1, rows); ++i) {
      for (int j = 0; j < cols; ++j) {
        buffer[j] = A[i][j];
      }
      // Send buffer to rank i + 1 with tag i.
      world.send(i + 1, i, buffer);
      cols_sent += 1;
    }
    // For each worker, recieve result, send rank more work
    for (int i = 0; i < rows; ++i) {
      s = world.recv(mpi::any_source, i, result);
      c[i] = result;
      if (cols_sent < rows) {
        // Allocate new buffer, and send more work
        for (int j = 0; j < rows; ++j) {
          buffer[j] = A[i][j];
        }
        world.send(s.source(), cols_sent, buffer);
        cols_sent += 1;
      }
      else {
        world.send(s.source(), env.max_tag(), buffer);
      }
    }
  }
  // Worker Code
  else {
    // Workers should continue running until they recieve the stop signal
    bool keepRunning{true};
    while (keepRunning) {
      // Get any_tag sending a buffer, the tag is the row
      s = world.recv(0, mpi::any_tag, buffer);
      row = s.tag();
      // If the tag is the max tag, stop running, otherwise do work and send
      // -> result back
      if (row == env.max_tag()) {
        keepRunning = false;
      }
      else {
        // Multiply the buffer with b, and send result
        result = 0;
        for (int j = 0; j < rows; ++j) {
          result += buffer[j] * b[j];
        }
        world.send(0, row, result);
      }
    }
  }

  // Print the elapsed time and c.
  if (world.rank() == 0) {
    cout << boost::format("-> Elapsed time %.16fs\n") % timer.elapsed();
    for (int j = 0; j < rows; ++j) {
      cout << boost::format("c[%i] = %.16f\n") % j % c[j];
    }
  }

  return 0;
}
