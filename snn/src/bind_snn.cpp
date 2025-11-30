#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "snn.hpp"

PYBIND11_MODULE(snn, m)
{
    m.doc() = "Spiking Neural Network";

    pybind11::class_<IzhikevichNeuron>(m, "IzhikevichNeuron")
        .def(pybind11::init<int, double>())
        .def("reset_state", &IzhikevichNeuron::reset_state)
        .def("update", &IzhikevichNeuron::update)
        .def("size", &IzhikevichNeuron::size)
        .def_readonly("v", &IzhikevichNeuron::v)
        .def_readonly("u", &IzhikevichNeuron::u);

    pybind11::class_<DoubleExponentialSynapticFilter>(m, "DoubleExponentialSynapticFilter")
        .def(pybind11::init<int, double>())
        .def("reset_state", &DoubleExponentialSynapticFilter::reset_state)
        .def("update", &DoubleExponentialSynapticFilter::update)
        .def("size", &DoubleExponentialSynapticFilter::size)
        .def_readonly("r", &DoubleExponentialSynapticFilter::r)
        .def_readonly("h", &DoubleExponentialSynapticFilter::h);

    pybind11::class_<SpikingNeuralNetwork>(m, "SpikingNeuralNetwork")
        .def(pybind11::init<int, int, int, double, double, double, double, double, double>())
        .def("reset_state", &SpikingNeuralNetwork::reset_state)
        .def("update", &SpikingNeuralNetwork::update)
        .def("train", &SpikingNeuralNetwork::train)
        .def("size", &SpikingNeuralNetwork::size)
        .def_readonly("x", &SpikingNeuralNetwork::x)
        .def_readonly("neurons", &SpikingNeuralNetwork::neurons)
        .def_readonly("synapses", &SpikingNeuralNetwork::synapses);
}

/*

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
extern "C"
{
#include "snn.h"
}

class _SpikingNeuralNetwork
{
public:
    Eigen::VectorXd x;
    Eigen::VectorXd r;
    Eigen::VectorXd h;
    Eigen::VectorXd v;
    Eigen::VectorXd u;
    struct SpikingNeuralNetwork data;
    _SpikingNeuralNetwork(int n_units, int in_size, int out_size, double dt, double connection_ratio, double G, double Q, double alpha, double bias);
    void reset_state();
    void update(Eigen::Ref<const Eigen::VectorXd> input);
    void train(Eigen::Ref<const Eigen::VectorXd> teaching_signal);
};

PYBIND11_MODULE(snn, m)
{
    m.doc() = "Spiking Neural Network";

    pybind11::class_<_SpikingNeuralNetwork>(m, "SpikingNeuralNetwork")
        .def(pybind11::init<int, int, int, double, double, double, double, double, double>())
        .def("reset_state", &_SpikingNeuralNetwork::reset_state)
        .def("update", &_SpikingNeuralNetwork::update)
        .def("train", &_SpikingNeuralNetwork::train)
        .def_readonly("x", &_SpikingNeuralNetwork::x)
        .def_readonly("v", &_SpikingNeuralNetwork::v)
        .def_readonly("u", &_SpikingNeuralNetwork::u)
        .def_readonly("r", &_SpikingNeuralNetwork::r)
        .def_readonly("h", &_SpikingNeuralNetwork::h);
}

_SpikingNeuralNetwork::_SpikingNeuralNetwork(int n_units, int in_size, int out_size, double dt, double connection_ratio, double G, double Q, double alpha, double bias)
{
    initialize_snn(&this->data, n_units, in_size, out_size, dt, connection_ratio, G, Q, alpha, bias);

    this->x = Eigen::Map<Eigen::VectorXd>(this->data.x, out_size, 1);
    this->v = Eigen::Map<Eigen::VectorXd>(this->data.v, n_units, 1);
    this->u = Eigen::Map<Eigen::VectorXd>(this->data.u, n_units, 1);
    this->r = Eigen::Map<Eigen::VectorXd>(this->data.r, n_units, 1);
    this->h = Eigen::Map<Eigen::VectorXd>(this->data.h, n_units, 1);
}

void _SpikingNeuralNetwork::reset_state()
{
    reset_snn(&this->data);
}

void _SpikingNeuralNetwork::update(Eigen::Ref<const Eigen::VectorXd> input)
{
    update_snn(&this->data, input.data());

    for (size_t i = 0; i < this->data.n_units; i++)
    {
        this->v[i] = this->data.v[i];
        this->u[i] = this->data.u[i];
        this->r[i] = this->data.r[i];
        this->h[i] = this->data.h[i];
    }
}

void _SpikingNeuralNetwork::train(Eigen::Ref<const Eigen::VectorXd> teaching_signal)
{
    train_snn(&this->data, teaching_signal.data());
}

*/