#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "synergy.hpp"

PYBIND11_MODULE(synergy, m)
{
    m.doc() = "Synergy";

    pybind11::class_<TimeVaryingSynergy>(m, "TimeVaryingSynergy")
        .def(pybind11::init<int, int, int>())
        .def("extract", &TimeVaryingSynergy::extract)
        .def("encode", &TimeVaryingSynergy::encode)
        .def("decode", &TimeVaryingSynergy::decode)
        .def_readonly("synergies", &TimeVaryingSynergy::synergies);
}
