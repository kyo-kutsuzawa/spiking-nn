#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "snn.hpp"

PYBIND11_MODULE(snn, m)
{
    m.doc() = "Spiking Neural Network";

    pybind11::class_<IzhikevichNeuron>(m, "IzhikevichNeuron")
        .def(pybind11::init<double, double>())
        .def("reset_state", &IzhikevichNeuron::reset_state)
        .def("update", &IzhikevichNeuron::update)
        .def("size", &IzhikevichNeuron::size)
        .def_readonly("v", &IzhikevichNeuron::v)
        .def_readonly("u", &IzhikevichNeuron::u);

    pybind11::class_<DoubleExponentialSynapticFilter>(m, "DoubleExponentialSynapticFilter")
        .def(pybind11::init<double, double>())
        .def("reset_state", &DoubleExponentialSynapticFilter::reset_state)
        .def("update", &DoubleExponentialSynapticFilter::update)
        .def("size", &DoubleExponentialSynapticFilter::size)
        .def_readonly("r", &DoubleExponentialSynapticFilter::r)
        .def_readonly("h", &DoubleExponentialSynapticFilter::h);

    pybind11::class_<SpikingNeuralNetwork>(m, "SpikingNeuralNetwork")
        .def(pybind11::init<double, double, double, double>())
        .def("reset_state", &SpikingNeuralNetwork::reset_state)
        .def("update", &SpikingNeuralNetwork::update)
        .def("train", &SpikingNeuralNetwork::train)
        .def("size", &SpikingNeuralNetwork::size)
        .def_readonly("x", &SpikingNeuralNetwork::x)
        .def_readonly("neurons", &SpikingNeuralNetwork::neurons)
        .def_readonly("synapses", &SpikingNeuralNetwork::synapses);
}