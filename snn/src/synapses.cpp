#include "snn.hpp"

DoubleExponentialSynapticFilter::DoubleExponentialSynapticFilter()
{
}

DoubleExponentialSynapticFilter::DoubleExponentialSynapticFilter(int n_units, double dt)
{
    this->n_units = n_units;
    this->dt = dt;
    this->tau_r = 2.0;
    this->tau_d = 20.0;

    this->r = Eigen::VectorXd(this->n_units);
    this->h = Eigen::VectorXd(this->n_units);
    this->reset_state();
}

void DoubleExponentialSynapticFilter::reset_state()
{
    int i;

    for (i = 0; i < this->n_units; i++)
    {
        this->r[i] = 0.0;
        this->h[i] = 0.0;
    }
}

void DoubleExponentialSynapticFilter::update(const Eigen::Ref<const Eigen::VectorXd> spikes)
{
    int i;

    for (i = 0; i < this->n_units; i++)
    {
        this->r[i] = (1 - this->dt / this->tau_d) * this->r[i] + this->h[i] * this->dt;
        this->h[i] = (1 - this->dt / this->tau_r) * this->h[i] + spikes[i] / (this->tau_r * this->tau_d);
    }
}

int DoubleExponentialSynapticFilter::size()
{
    return this->n_units;
}
