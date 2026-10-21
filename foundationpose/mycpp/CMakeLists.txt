cmake_minimum_required(VERSION 3.19...3.31)
project(mycpp LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost 1.82 CONFIG REQUIRED)
find_package(Eigen3 3.4 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB MY_SRC CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)

pybind11_add_module(mycpp src/app/pybind_api.cpp ${MY_SRC})
target_compile_features(mycpp PRIVATE cxx_std_17)
target_include_directories(mycpp PRIVATE include)
target_link_libraries(
  mycpp
  PRIVATE
    Boost::headers
    Eigen3::Eigen
)
