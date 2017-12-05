#include <boost/multi_array.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/timer.hpp>
#include <boost/format.hpp>

namespace boost {
namespace serialization {
    template <class Archive, class T, size_t N>
    void serialize(Archive &ar, multi_array<T, N> &ma, const unsigned int version) {
        for (auto i = 0; i < ma.num_elements(); ++i) {
            ar & ma.data()[i];
        }
    }
} // namespace serialization
} // namespace boost

namespace mpi = boost::mpi;

int main(void) {
    mpi::environment env;
    mpi::communicator world;

    const int rows{10}, cols{10};
    mpi::status status;
    double result{0.0};
    int row{0};

    using boost::multi_array;
    using boost::extents;
    multi_array<double, 1> b(extents[cols]), c(extents[cols]), buffer(extents[cols]) ;
    multi_array<double, 2> A(extents[rows][cols]);

    // Start the Timer
    mpi::timer timer;

    // Initialize on rank 0, broadcast column vector
    if (world.rank() == 0) {
        for (auto i = 0; i < cols; ++i) {
            b[i] = i;
            c[i] = 0.0;
            for (auto j = 0; j < rows; ++j) {
                A[i][j] = i;
            }
        }
    }
    mpi::broadcast(world, b, 0);

    // Manager Code
    if (world.rank() == 0) {
        int columns_sent = 0;

        // Send the initial buffers to be worked on
        for (auto i = 0; i < std::min(world.size() - 1, rows); ++i) {
            for (auto j = 0; j < cols; ++j) {
                buffer[j] = A[i][j];
            }
            world.send(i + 1, i, buffer);
            columns_sent += 1;
        }

        // Receive the workers result, send more work if necessary
        for (auto i = 0; i < rows; ++i) {
            status = world.recv(mpi::any_source, i, result);
            c[i] = result;
            // If the number of columns_sent is less than rows, send the next row
            if (columns_sent < rows) {
                for (auto j = 0; j < rows; ++j) {
                    // Sending the next row, i.e. columns_sent
                    buffer[j] = A[columns_sent][j];
                }
                world.send(status.source(), columns_sent, buffer);
                columns_sent += 1;
            }
            else {
                world.send(status.source(), env.max_tag(), buffer);
            }
        }
    }
    // Worker code
    else {
        bool keep_running{true};
        while (keep_running) {
            // Get any tag sending buffer, tag is the row (used to determine end condition)
            status = world.recv(0, mpi::any_tag, buffer);
            row = status.tag();

            if (row == env.max_tag()) {
                keep_running = false;
            }
            else {
                result = 0;
                for (auto j = 0; j < rows; ++j) {
                    result += buffer[j] * b[j];
                }
                world.send(0, row, result);
            }
        }
    }

    // Elapsed Time & Result
    if (world.rank() == 0) {
        std::cout << boost::format("-> Elapsed time %.16fs\n") % timer.elapsed();
        for (auto j = 0; j < rows; ++j) std::cout << boost::format("c[%2i] = %.16f\n") % j % c[j];
    }
}
