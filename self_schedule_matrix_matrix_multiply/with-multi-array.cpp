#include <boost/multi_array.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/timer.hpp>
#include <boost/format.hpp>
#include <string>

typedef boost::multi_array<double, 2> matrix_2_d;
typedef matrix_2_d::index matrix_2_d_index;

namespace boost {
namespace serialization {
    template <class Archive>
    void serialize(Archive &ar, matrix_2_d &m, const unsigned int version) {
        for (auto i = 0; i < m.num_elements(); ++i) {
            ar & m.data()[i];
        }
    }
} // namespace serialization
} // namespace boost

void matrix_print(const std::string &label, const matrix_2_d &m) {
    for (auto i = 0; i < m.shape()[0]; ++i) {
        std::cout << boost::format("%s[%2i][:] = ") % label % i;
        for (auto j = 0; j < m.shape()[1]; ++j) {
            if (j == m.shape()[1] - 1) {
                std::cout << boost::format("%.16f") % m[i][j];
            }
            else {
                std::cout << boost::format("%.16f, ") % m[i][j];
            }
        }
        std::cout << '\n';
    }
}

namespace mpi = boost::mpi;

int main(void) {
    mpi::environment env;
    mpi::communicator world;

    const int rows{10}, cols{10};
    mpi::status status;
    double result{0};
    int item{0};
    matrix_2_d A(boost::extents[rows][cols]), B(boost::extents[rows][cols]), C(boost::extents[rows][cols]);
    matrix_2_d buffer(boost::extents[2][cols]);

    // Start the Timer
    mpi::timer timer;

    // Initialize on rank 0, broadcast column vector
    if (world.rank() == 0) {
        for (auto i = 0; i < cols; ++i) {
            for (auto j = 0; j < rows; ++j) {
                A[i][j] = i;
                B[i][j] = j;
                C[i][j] = 0.0;
            }
        }
    }

    // Manager Code
    if (world.rank() == 0) {
        int items_sent = 0;

        // Send the initial buffers to be worked on
        for (auto i = 0; i < std::min(world.size() - 1, rows); ++i) {
            for (auto j = 0; j < std::min(world.size() - 1, cols); ++j) {
                for (auto k = 0; k < rows; ++k) {
                    buffer[0][k] = A[i][k];
                    buffer[1][k] = B[k][j];
                }
                world.send(i + 1, (rows * i + j), buffer);
                items_sent += 1;
            }
        }

        // Receive the workers result, send more work if necessary
        for (auto i = 0; i < rows; ++i) {
            for (auto j = 0; j < cols; ++j) {
                status = world.recv(mpi::any_source, rows * i + j, result);
                C[i][j] = result;
                // If the number of columns_sent is less than rows, send the next row
                if (items_sent < (rows * cols)) {
                    // Send i + 1 row w/ j = 0 col
                    if (j ==  cols - 1) {
                        for (auto k = 0; k < rows; ++k) {
                            buffer[0][k] = A[i + 1][k];
                            buffer[1][k] = B[k][0];
                        }
                        world.send(status.source(), rows * (i + 1) + 0, buffer);
                        items_sent += 1;
                    }
                    // Else, send i row w/ j + 1 col
                    else {
                        for (auto k = 0; k < rows; ++k) {
                            buffer[0][k] = A[i][k];
                            buffer[1][k] = B[k][j + 1];
                        }
                        world.send(status.source(), rows * i + (j + 1), buffer);
                        items_sent += 1;
                    }
                } else {
                    world.send(status.source(), env.max_tag(), buffer);
                }
            }
        }
    }
    // Worker code
    else {
        bool keep_running{true};
        while (keep_running) {
            // Get any tag sending buffer, tag is the row * i + j (used to determine end condition)
            status = world.recv(0, mpi::any_tag, buffer);
            item = status.tag();

            if (item == env.max_tag()) {
                keep_running = false;
            }
            else {
                result = 0;
                for (auto k = 0; k < rows; ++k) {
                    result += buffer[0][k] * buffer[1][k];
                }
                world.send(0, item, result);
            }
        }
    }

    // Wait for everything to finish
    world.barrier();

    // Elapsed Time & Result
    if (world.rank() == 0) {
        std::cout << boost::format("-> Elapsed time %.16fs\n") % timer.elapsed();
        matrix_print("A", A);
        std::cout << "\n";
        matrix_print("B", B);
        std::cout << "\n";
        matrix_print("C", C);
    }
}
