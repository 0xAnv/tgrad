# min versions
cmake_minimum_required(VERSION 3.14)
project(tgrad VERSION 0.1.0 LANGUAGES CXX)

# set c++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # disables compiler specific extensions; keeps code standard

# define core library
add_library(tgrad
        src/dtype.cpp
        src/device.cpp
        src/storage.cpp
        src/shape.cpp
        src/tensor.cpp
)

# define where header file lives
target_include_directories(tgrad PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_executable(test_tensor tests/unit/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE tgrad)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tgrad PRIVATE -Wall -Wextra -Wpedantic -O3)
endif ()
