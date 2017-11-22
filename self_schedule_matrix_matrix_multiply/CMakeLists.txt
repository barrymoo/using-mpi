cmake_minimum_required(VERSION 3.6)
project(self_schedule_matrix_matrix_multiply)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_COMPILER "mpicxx")

find_package(MPI REQUIRED)
find_package(Boost COMPONENTS mpi serialization REQUIRED)

set(WITH_MA_FILES with-multi-array.cpp)
add_executable(with-ma ${WITH_MA_FILES})

target_include_directories(with-ma PUBLIC
        ${MPI_INCLUDE_PATH}
        ${Boost_INCLUDE_DIRS}
)

target_link_libraries(with-ma ${Boost_LIBRARIES})
