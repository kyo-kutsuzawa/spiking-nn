#include <stdlib.h>
#include <time.h>
#include "snn.h"

void initialize_izhikevich_neurons(struct IzhikevichNeurons *neurons, int n_units, double dt)
{
    neurons->v = malloc(n_units * sizeof(double));
    neurons->u = malloc(n_units * sizeof(double));

    neurons->n_units = n_units;
    neurons->dt = dt;

    neurons->C = 250.0;
    neurons->k = 2.5;
    neurons->a = 0.01;
    neurons->b = -2.0;
    neurons->d = 200.0;
    neurons->vr = -60.0;
    neurons->vt = neurons->vr + 40.0 - neurons->b / neurons->k;
    neurons->v_peak = 30.0;
    neurons->v_reset = -65.0;

    neurons->dt_c = neurons->dt / neurons->C;
    neurons->dt_a = neurons->dt * neurons->a;
}

void reset_izhikevich_neurons(struct IzhikevichNeurons *neurons)
{
    double noise;
    int i;

    srand((unsigned int)clock());

    for (i = 0; i < neurons->n_units; i++)
    {
        noise = (double)rand() / (double)RAND_MAX;
        neurons->v[i] = 0.0;
        neurons->u[i] = neurons->vr + (neurons->v_peak - neurons->vr) + noise;
    }
}

void update_izhikevich_neurons(struct IzhikevichNeurons *neurons, double *spikes, const double *input)
{
    double vi;
    int i;

    for (i = 0; i < neurons->n_units; i++)
    {
        vi = neurons->v[i];
        neurons->v[i] += neurons->dt_c * (neurons->k * (neurons->v[i] - neurons->vr) * (neurons->v[i] - neurons->vt) - neurons->u[i] + input[i]);
        neurons->u[i] += neurons->dt_a * (neurons->b * (vi - neurons->vr) - neurons->u[i]);

        if (neurons->v[i] >= neurons->v_peak)
        {
            spikes[i] = 1.0;
            neurons->u[i] += neurons->d;
            neurons->v[i] = neurons->v_reset;
        }
        else
        {
            spikes[i] = 0.0;
        }
    }
}
