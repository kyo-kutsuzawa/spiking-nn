#include <random>
#include <vector>
#include "snn.hpp"

IzhikevichNeuron::IzhikevichNeuron()
{
}

IzhikevichNeuron::IzhikevichNeuron(int n_units, double dt)
{
    this->n_units = n_units;
    this->dt = dt;
    this->C = 250.0;
    this->k = 2.5;
    this->a = 0.01;
    this->b = -2.0;
    this->d = 200.0;

    this->vr = -60.0;
    this->vt = this->vr + 40.0 - this->b / this->k;
    this->v_peak = 30.0;
    this->v_reset = -65.0;

    // this->rand_engine = std::default_random_engine(rd());
    // this->dist = std::uniform_real_distribution<double>(0.0, 1.0);

    this->v = Eigen::VectorXd(this->n_units);
    this->u = Eigen::VectorXd(this->n_units);
    this->reset_state();
}

void IzhikevichNeuron::reset_state()
{
    std::random_device rd;
    std::default_random_engine rand_engine(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double noise;
    int i;

    for (i = 0; i < this->n_units; i++)
    {
        noise = dist(rand_engine);
        this->v[i] = 0.0;
        this->u[i] = this->vr + (this->v_peak - this->vr) + noise;
    }
}

Eigen::VectorXd IzhikevichNeuron::update(Eigen::Ref<const Eigen::VectorXd> input)
{
    Eigen::VectorXd spikes = Eigen::VectorXd::Zero(this->n_units);
    double vi;
    int i;

    for (i = 0; i < this->n_units; i++)
    {
        vi = this->v[i];
        this->v[i] += (this->dt / this->C * (this->k * (this->v[i] - this->vr) * (this->v[i] - this->vt) - this->u[i] + input[i]));
        this->u[i] += this->dt * this->a * (this->b * (vi - this->vr) - this->u[i]);

        if (this->v[i] >= this->v_peak)
        {
            spikes[i] = 1.0;
            this->u[i] += this->d;
            this->v[i] = this->v_reset;
        }
    }

    return spikes;
}

int IzhikevichNeuron::size()
{
    return this->n_units;
}
