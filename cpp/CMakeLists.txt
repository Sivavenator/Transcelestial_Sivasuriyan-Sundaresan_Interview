cmake_minimum_required(VERSION 3.16)
project(sptrack_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -O3 with autovectorisation, but deliberately NOT -ffast-math.
#
# -ffast-math permits reassociation of floating-point sums, which changes
# summation order. The centroid is cross-validated against the Python
# reference to 1e-9 px, and that tolerance assumes only ordinary rounding
# differences. Enabling fast-math would silently loosen the numerics the
# validation depends on. If it is enabled for performance, re-run
# test_against_python first and confirm the tolerances still hold on the
# target compiler.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-O3 -Wall -Wextra -Wpedantic)
elseif(MSVC)
  add_compile_options(/O2 /W4)
endif()

add_library(sptrack_estimators INTERFACE)
target_include_directories(sptrack_estimators
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_against_python tests/test_against_python.cpp)
target_link_libraries(test_against_python PRIVATE sptrack_estimators)

add_executable(bench src/bench.cpp)
target_link_libraries(bench PRIVATE sptrack_estimators)

enable_testing()

# The vector file is produced by the Python side:
#   python -m tools.export_cpp_vectors
# It is committed, so a fresh clone can run ctest without running Python
# first, but regenerating it after any change to the Python estimators is
# what keeps the cross-validation meaningful.
add_test(NAME cross_validation_against_python
         COMMAND test_against_python
                 ${CMAKE_CURRENT_SOURCE_DIR}/../results/cpp_vectors.csv)
