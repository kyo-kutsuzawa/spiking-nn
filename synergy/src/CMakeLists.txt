cmake_minimum_required(VERSION 4.0)

project(synergy)
set(PYBIND11_CPP_STANDARD -std=c++11)
set(CMAKE_CXX_FLAGS "-W4 -O2 -EHsc")
add_subdirectory(../../pybind11 pybind11)
include_directories(../../eigen-5.0.0)

if(MSVC)
    add_compile_options("/utf-8")
endif()

pybind11_add_module(synergy bind_synergy.cpp synergy.cpp)
