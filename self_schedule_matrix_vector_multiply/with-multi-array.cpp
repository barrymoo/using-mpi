#include <boost/multi_array.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/timer.hpp>
#include <boost/format.hpp>

typedef boost::multi_array<double, 1> vector_d;
typedef vector_d::index vector_d_index;

typedef boost::multi_array<double, 2> matrix_2_d;
typedef matrix_2_d::index matrix_2_d_index;

namespace boost {
namespace serialization {
    template <class Archive>
    void serialize(Archive &ar, vector_d &v, const unsigned int version) {
        for (auto i = 0; i < v.num_elements(); ++i) {
            ar & v.data()[i];
        }
    }

//    template <class Archive>
//    void serialize(Archive &ar, matrix_2_d &m, const unsigned int version) {
//        for (auto i = 0; i < m.num_elements(); ++i) {
//            ar & m.data()[i];
//        }
//    }
} // namespace serialization
} // namespace boost

namespace mpi = boost::mpi;

int main(void) {
    mpi::environment env;
    mpi::communicator world;

    const int rows{10}, cols{10};
    mpi::status status;
    double result{0};
    int row{0};
    vector_d b(boost::extents[cols]), c(boost::extents[cols]), buffer(boost::extents[cols]) ;
    matrix_2_d A(boost::extents[rows][cols]);

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
