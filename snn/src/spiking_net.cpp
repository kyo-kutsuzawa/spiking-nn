#include <cmath>
#include <iostream>
#include <vector>
#include "snn.hpp"

SpikingNeuralNetwork::SpikingNeuralNetwork(int n_units, int in_size, int out_size, double dt, double connection_ratio, double G, double Q, double alpha, double bias)
{
    std::random_device rd;
    std::default_random_engine rand_engine(rd());
    std::uniform_real_distribution<double> dist_01(0.0, 1.0);
    std::uniform_real_distribution<double> dist_11(-1.0, 1.0);
    std::normal_distribution<double> dist_normal(0.0, 1.0);
    std::uniform_int_distribution<int> dist_idx(0, n_units - 1);
    std::vector<Eigen::Triplet<double>> triplet_vec;
    double coef;
    int i, j;

    this->n_units = n_units;
    this->in_size = in_size;
    this->out_size = out_size;
    this->dt = dt;

    this->p = connection_ratio;
    this->G = G;
    this->Q = Q;
    this->alpha = alpha;

    this->phi = RowMatrixXd::Zero(out_size, n_units);

    this->eta = RowMatrixXd(n_units, out_size);
    for (i = 0; i < n_units; i++)
    {
        for (j = 0; j < out_size; j++)
        {
            this->eta(i, j) = dist_11(rand_engine);
        }
    }

    this->w0 = RowMatrixXd::Zero(n_units, n_units);
    coef = 1.0 / (sqrt(this->n_units) * this->p);
    for (i = 0; i < n_units; i++)
    {
        for (j = 0; j < n_units; j++)
        {
            if (dist_01(rand_engine) < this->p)
            {
                this->w0(i, j) = dist_normal(rand_engine) * coef;
            }
        }
    }

    this->w0_sp = Eigen::SparseMatrix<double>(n_units, n_units);
    for (i = 0; i < (int)(n_units * n_units * connection_ratio); i++)
    {
        triplet_vec.push_back(Eigen::Triplet<double>(dist_idx(rand_engine), dist_idx(rand_engine), dist_normal(rand_engine) * coef));
    }
    this->w0_sp.setFromTriplets(triplet_vec.begin(), triplet_vec.end());

    this->i_bias = Eigen::VectorXd(n_units);
    for (i = 0; i < n_units; i++)
    {
        this->i_bias[i] = bias;
    }

    this->P = RowMatrixXd::Identity(n_units, n_units);
    for (i = 0; i < n_units; i++)
    {
        this->P(i, i) /= alpha;
    }

    this->Gw0 = G * this->w0;
    this->Qeta = Q * this->eta;
    this->errors = Eigen::VectorXd(out_size);
    this->Pr = Eigen::VectorXd(n_units);
    this->PrrP = RowMatrixXd(n_units, n_units);
    this->Gw0_sp = G * this->w0_sp;

    this->current = Eigen::VectorXd(n_units);
    this->spikes = Eigen::VectorXd(n_units);

    this->x = Eigen::VectorXd(out_size);
    this->neurons = IzhikevichNeuron(n_units, dt);
    this->synapses = DoubleExponentialSynapticFilter(n_units, dt);
    this->reset_state();
}

void SpikingNeuralNetwork::reset_state()
{
    int i;

    for (i = 0; i < this->out_size; i++)
    {
        this->x[i] = 0.0;
    }

    this->neurons.reset_state();
    this->synapses.reset_state();
}

void SpikingNeuralNetwork::update(const Eigen::Ref<const Eigen::VectorXd> input)
{
    int i;

    // Calculate input currents
    this->current = this->Gw0_sp * this->synapses.r + this->Qeta * this->x + this->i_bias + input;

    for (i = 0; i < this->n_units; i++)
    {
        this->spikes[i] = 0.0;
    }

    // Update the states of neurons and synapses
    this->neurons.update(this->spikes, this->current);
    this->synapses.update(this->spikes);

    this->x = this->phi * this->synapses.r;
}

void SpikingNeuralNetwork::train(const Eigen::Ref<const Eigen::VectorXd> teaching_signal)
{
    double rPr;

    this->errors = this->x - teaching_signal;

    // Update P
    this->Pr = this->P * this->synapses.r;
    rPr = this->synapses.r.dot(Pr);
    this->PrrP = Pr * Pr.transpose();
    this->P -= PrrP / (1.0 + rPr);

    // Update phi
    this->phi -= errors * Pr.transpose();
}

int SpikingNeuralNetwork::size()
{
    return this->n_units;
}
