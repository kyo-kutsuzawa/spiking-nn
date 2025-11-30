#include <stdlib.h>
#include "snn.h"

void initialize_double_exponential_synaptic_filters(struct DoubleExponentialSynapticFilters *synapses, int n_units, double dt)
{
    synapses->r = malloc(n_units * sizeof(double));
    synapses->h = malloc(n_units * sizeof(double));

    synapses->n_units = n_units;
    synapses->dt = dt;

    synapses->tau_r = 2.0;
    synapses->tau_d = 20.0;

    synapses->gr_dt = 1.0 - synapses->dt / synapses->tau_r;
    synapses->gd_dt = 1.0 - synapses->dt / synapses->tau_d;
    synapses->g_rd = 1.0 / (synapses->tau_r * synapses->tau_d);
}

void reset_double_exponential_synaptic_filters(struct DoubleExponentialSynapticFilters *synapses)
{
    int i;

    for (i = 0; i < synapses->n_units; i++)
    {
        synapses->r[i] = 0.0;
        synapses->h[i] = 0.0;
    }
}

void update_double_exponential_synaptic_filters(struct DoubleExponentialSynapticFilters *synapses, const double *spikes)
{
    int i;

    for (i = 0; i < synapses->n_units; i++)
    {
        synapses->r[i] = synapses->gd_dt * synapses->r[i] + synapses->h[i] * synapses->dt;
        synapses->h[i] = synapses->gr_dt * synapses->h[i] + spikes[i] * synapses->g_rd;
    }
}
