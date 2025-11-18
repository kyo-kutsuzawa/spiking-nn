#include <cmath>
#include <iostream>
#include "snn.hpp"

SpikingNeuralNetwork::SpikingNeuralNetwork(int n_units, int in_size, int out_size, double dt)
{
    std::random_device rd;
    std::default_random_engine rand_engine(rd());
    std::uniform_real_distribution<double> dist_01(0.0, 1.0);
    std::uniform_real_distribution<double> dist_11(-1.0, 1.0);
    std::normal_distribution<double> dist_normal(0.0, 1.0);

    this->n_units = n_units;
    this->in_size = in_size;
    this->out_size = out_size;
    this->dt = dt;

    this->p = 0.1;
    this->G = 5e3;
    this->Q = 5e3;
    this->l = 2.0;

    this->phi = RowMatrixXd::Zero(out_size, n_units);
    this->eta = RowMatrixXd(n_units, out_size);
    for (int i = 0; i < n_units; i++)
    {
        for (int j = 0; j < out_size; j++)
        {
            this->eta(i, j) = dist_11(rand_engine);
        }
    }

    this->w0 = RowMatrixXd::Zero(n_units, n_units);
    double coef = 1.0 / (sqrt(this->n_units) * this->p);
    for (int i = 0; i < n_units; i++)
    {
        for (int j = 0; j < n_units; j++)
        {
            if (dist_01(rand_engine) < this->p)
            {
                this->w0(i, j) = dist_normal(rand_engine) * coef;
            }
        }
    }

    this->i_bias = Eigen::VectorXd(n_units);
    for (int i = 0; i < n_units; i++)
    {
        this->i_bias[i] = 1000.0;
    }
    this->P = RowMatrixXd::Identity(n_units, n_units);
    for (int i = 0; i < n_units; i++)
    {
        this->P(i, i) /= l;
    }

    this->Gw0 = G * this->w0;
    this->Qeta = Q * this->eta;

    this->x = Eigen::VectorXd(n_units);
    this->neurons = IzhikevichNeuron(n_units, dt);
    this->synapses = DoubleExponentialSynapticFilter(n_units, dt);
    this->reset_state();
}

void SpikingNeuralNetwork::reset_state()
{
    for (int i = 0; i < this->n_units; i++)
    {
        this->x[i] = 0.0;
    }

    this->neurons.reset_state();
    this->synapses.reset_state();
}

void SpikingNeuralNetwork::update(Eigen::Ref<const Eigen::VectorXd> input)
{
    // Calculate input currents
    Eigen::VectorXd current = this->Gw0 * this->synapses.r + this->Qeta * this->x + this->i_bias;

    // Update the states of neurons and synapses
    Eigen::VectorXd spikes = this->neurons.update(current);
    this->synapses.update(spikes);

    this->x = this->phi * this->synapses.r;
}

void SpikingNeuralNetwork::train(Eigen::Ref<const Eigen::VectorXd> teaching_signal)
{
    Eigen::VectorXd errors = this->x - teaching_signal;

    // Update P
    Eigen::VectorXd Pr = this->P * this->synapses.r;
    double rPr = this->synapses.r.dot(Pr);
    RowMatrixXd rPPr = Pr * Pr.transpose();
    this->P -= rPPr / (1.0 + rPr);

    // Update phi
    this->phi -= errors * Pr.transpose();
}

int SpikingNeuralNetwork::size()
{
    return this->n_units;
}