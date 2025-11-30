#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "snn.h"

#include <stdio.h>

#define M_PI 3.1415926536

double rand_01()
{
    return (double)rand() / (double)RAND_MAX;
}

double rand_11()
{
    return ((double)rand() * 2.0) / (double)RAND_MAX - 1.0;
}

double rand_normal()
{
    return sqrt(-2.0 * log(rand_01())) * sin(2.0 * M_PI * rand_01());
}

void initialize_snn(struct SpikingNeuralNetwork *snn, int n_units, int in_size, int out_size, double dt, double connection_ratio, double G, double Q, double alpha, double bias)
{
    double coef;
    int i, j;

    snn->n_units = n_units;
    snn->in_size = in_size;
    snn->out_size = out_size;
    snn->dt = dt;

    snn->p = connection_ratio;
    snn->G = G;
    snn->Q = Q;
    snn->l = alpha;

    snn->phi = (double *)calloc(out_size * n_units, sizeof(double));
    snn->eta = (double *)malloc(n_units * out_size * sizeof(double));
    snn->Qeta = (double *)malloc(n_units * out_size * sizeof(double));
    for (i = 0; i < n_units; i++)
    {
        for (j = 0; j < out_size; j++)
        {
            snn->eta[out_size * i + j] = rand_11();
            snn->Qeta[out_size * i + j] = snn->Q * snn->eta[out_size * i + j];
        }
    }

    snn->w0 = (double *)calloc(n_units * n_units, sizeof(double));
    snn->Gw0 = (double *)calloc(n_units * n_units, sizeof(double));
    coef = 1.0 / (sqrt(snn->n_units) * snn->p);
    for (i = 0; i < n_units; i++)
    {
        for (j = 0; j < n_units; j++)
        {
            if (rand_01() < snn->p)
            {
                snn->w0[n_units * i + j] = rand_normal() * coef;
                snn->Gw0[n_units * i + j] = snn->G * snn->w0[n_units * i + j];
            }
        }
    }

    snn->i_bias = (double *)malloc(n_units * sizeof(double));
    for (i = 0; i < n_units; i++)
    {
        snn->i_bias[i] = bias;
    }
    snn->P = (double *)calloc(n_units * n_units, sizeof(double));
    snn->Pr = (double *)malloc(n_units * sizeof(double));
    snn->PrrP = (double *)malloc(n_units * n_units * sizeof(double));
    for (i = 0; i < n_units; i++)
    {
        snn->P[n_units * i + i] = 1.0 / snn->l;
    }

    snn->x = (double *)malloc(out_size * sizeof(double));
    snn->spikes = (double *)malloc(n_units * sizeof(double));
    snn->current = (double *)malloc(n_units * sizeof(double));

    /* Neurons */
    snn->v = (double *)malloc(n_units * sizeof(double));
    snn->u = (double *)malloc(n_units * sizeof(double));

    snn->C = 250.0;
    snn->k = 2.5;
    snn->a = 0.01;
    snn->b = -2.0;
    snn->d = 200.0;
    snn->vr = -60.0;
    snn->vt = snn->vr + 40.0 - snn->b / snn->k;
    snn->v_peak = 30.0;
    snn->v_reset = -65.0;

    snn->dt_c = snn->dt / snn->C;
    snn->dt_a = snn->dt * snn->a;

    /* Synapses */
    snn->r = (double *)malloc(n_units * sizeof(double));
    snn->h = (double *)malloc(n_units * sizeof(double));

    snn->tau_r = 2.0;
    snn->tau_d = 20.0;

    snn->gr_dt = 1.0 - snn->dt / snn->tau_r;
    snn->gd_dt = 1.0 - snn->dt / snn->tau_d;
    snn->g_rd = 1.0 / (snn->tau_r * snn->tau_d);

    // initialize_izhikevich_neurons(&snn->neurons, n_units, dt);
    // initialize_double_exponential_synaptic_filters(&snn->synapses, n_units, dt);
}

void reset_snn(struct SpikingNeuralNetwork *snn)
{
    int i;

    for (i = 0; i < snn->out_size; i++)
    {
        snn->x[i] = 0.0;
    }

    for (i = 0; i < snn->n_units; i++)
    {
        snn->v[i] = 0.0;
        snn->u[i] = snn->vr + (snn->v_peak - snn->vr) + rand_01();

        snn->r[i] = 0.0;
        snn->h[i] = 0.0;
    }

    // reset_izhikevich_neurons(&snn->neurons);
    // reset_double_exponential_synaptic_filters(&snn->synapses);
}

void update_snn(struct SpikingNeuralNetwork *snn, const double *input)
{
    double vi;
    int i, j;

    for (i = 0; i < snn->n_units; i++)
    {
        // Calculate input currents
        snn->current[i] = input[i] + snn->i_bias[i];
        for (j = 0; j < snn->n_units; j++)
        {
            snn->current[i] += snn->Gw0[snn->n_units * i + j] * snn->r[j];
        }
        for (j = 0; j < snn->out_size; j++)
        {
            snn->current[i] += snn->Qeta[snn->out_size * i + j] * snn->x[j];
        }

        /* Neurons */
        vi = snn->v[i];
        snn->v[i] += snn->dt_c * (snn->k * (snn->v[i] - snn->vr) * (snn->v[i] - snn->vt) - snn->u[i] + input[i]);
        snn->u[i] += snn->dt_a * (snn->b * (vi - snn->vr) - snn->u[i]);
        snn->spikes[i] = 0.0;

        if (snn->v[i] >= snn->v_peak)
        {
            snn->spikes[i] = 1.0;
            snn->u[i] += snn->d;
            snn->v[i] = snn->v_reset;
        }

        /* Synapses */
        snn->r[i] = snn->gd_dt * snn->r[i] + snn->h[i] * snn->dt;
        snn->h[i] = snn->gr_dt * snn->h[i] + snn->spikes[i] * snn->g_rd;
    }

    // Update the states of neurons and synapses
    // update_izhikevich_neurons(&snn->neurons, snn->spikes, snn->current);
    // update_double_exponential_synaptic_filters(&snn->synapses, snn->spikes);

    // Calculate output values
    for (i = 0; i < snn->out_size; i++)
    {
        for (j = 0; j < snn->n_units; j++)
        {
            snn->x[i] = snn->phi[snn->n_units * i + j] * snn->r[j];
        }
    }
}

void train_snn(struct SpikingNeuralNetwork *snn, const double *teaching_signal)
{
    double error_i;
    double rPr;
    double coef;
    int i, j;

    // Calculate Pr
    for (i = 0; i < snn->n_units; i++)
    {
        for (j = 0; j < snn->n_units; j++)
        {
            snn->Pr[i] = snn->P[snn->n_units * i + j] * snn->r[j];
        }
    }

    // Calculate rPr
    // rPr = r[i] * P[i, j] * r[j] = r[i] * Pr[i]
    rPr = 0.0;
    for (i = 0; i < snn->n_units; i++)
    {
        rPr += snn->r[i] * snn->Pr[i];
    }
    coef = 1.0 / (1.0 + rPr);

    // Calculate PrrP
    // PrrP[i, j] = P[i, k] * r[k] * r[l] * P[l, j] = Pr[i] * Pr[j]
    for (i = 0; i < snn->out_size; i++)
    {
        snn->PrrP[i] = 0.0;
        for (j = 0; j < snn->n_units; j++)
        {
            snn->PrrP[snn->n_units * i + j] += snn->Pr[i] * snn->Pr[j];
        }
    }

    // Update phi (output connections)
    // phi[i, j] -= e[i] * Pr[j]
    for (i = 0; i < snn->out_size; i++)
    {
        error_i = snn->x[i] - teaching_signal[i];
        for (j = 0; j < snn->n_units; j++)
        {
            snn->phi[snn->n_units * i + j] -= error_i * snn->Pr[j];
        }
    }

    // Update P
    // P[i, j] -= (1 - rPr)^{-1} PrrP[i, j]
    for (i = 0; i < snn->n_units; i++)
    {
        for (j = 0; j < snn->n_units; j++)
        {
            snn->P[snn->n_units * i + j] -= coef * snn->PrrP[snn->n_units * i + j];
        }
    }
}
