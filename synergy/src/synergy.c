#include "synergy.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double rand_uniform01(void)
{
    return (double)rand() / RAND_MAX;
}

int max_int(int a, int b)
{
    if (a > b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

int min_int(int a, int b)
{
    if (a < b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

int initialize_synergies(struct TimeVaryingSynergy *synergies, int n_synergies, int synergy_length, int n_dim, int refractory_period)
{
    int i;

    synergies->synergies = (double *)malloc(n_synergies * synergy_length * n_dim * sizeof(double));
    synergies->n_synergies = n_synergies;
    synergies->synergy_length = synergy_length;
    synergies->n_dim = n_dim;
    synergies->refractory_period = refractory_period;

    // Initialize synergies with random non-negative values
    for (i = 0; i < n_synergies * synergy_length * n_dim; i++)
    {
        synergies->synergies[i] = rand_uniform01();
    }

    // Normalize synergies
    normalize_synergies(synergies);

    // Return -1 if memory allocation failed
    if (synergies->synergies == NULL)
    {
        return -1;
    }

    return 0;
}

int finalize_synergies(struct TimeVaryingSynergy *synergies)
{
    if (synergies->synergies != NULL)
    {
        free(synergies->synergies);
        synergies->synergies = NULL;
    }

    synergies->n_synergies = 0;
    synergies->synergy_length = 0;
    synergies->n_dim = 0;
    synergies->refractory_period = 0;

    return 0;
}

int initialize_activities(struct TVSynergyActivities *activities, int n_synergies, int n_activities_max)
{
    activities->amplitudes = (double *)calloc(n_synergies * n_activities_max, sizeof(double));
    activities->delays = (int *)calloc(n_synergies * n_activities_max, sizeof(int));
    activities->n_synergies = n_synergies;
    activities->n_activities_max = n_activities_max;

    // Return -1 if memory allocation failed
    if ((activities->amplitudes == NULL) || (activities->delays == NULL))
    {
        return -1;
    }

    return 0;
}

int finalize_activities(struct TVSynergyActivities *activities)
{
    if (activities->amplitudes != NULL)
    {
        free(activities->amplitudes);
        activities->amplitudes = NULL;
    }

    if (activities->delays != NULL)
    {
        free(activities->delays);
        activities->delays = NULL;
    }

    activities->n_synergies = 0;
    activities->n_activities_max = 0;

    return 0;
}

int extract(struct TimeVaryingSynergy *synergies, const double *trajectories, int n_data, int trajectory_length, int n_dim, int n_iter, double lr, int n_activities_max, int print_progress)
{
    struct TVSynergyActivities activities;
    double *gradient;
    double *trajectory;
    double *trajectory_reconstructed;
    int synergies_size;
    int trajectory_size;
    int ret_val;
    // int i, j, k, l;
    int i;
    int iter;

    srand((unsigned int)time(NULL));

    n_dim = synergies->n_dim;
    synergies_size = synergies->n_synergies * synergies->synergy_length * synergies->n_dim;
    trajectory_size = trajectory_length * n_dim;

    // Initialize synergy activities
    ret_val = initialize_activities(&activities, synergies->n_synergies, n_activities_max);

    // Initialize array variables
    gradient = (double *)malloc(synergies_size * sizeof(double));
    trajectory_reconstructed = (double *)calloc(trajectory_size, sizeof(double));

    // Return -1 if memory allocation failed
    if ((gradient == NULL) || (trajectory_reconstructed == NULL) || (ret_val == -1))
    {
        return -1;
    }

    for (iter = 0; iter < n_iter; iter++)
    {
        // Reset the gradient
        for (i = 0; i < synergies_size; i++)
        {
            gradient[i] = 0.0;
        }

        // Compute gradient for each trajectory
        for (i = 0; i < n_data; i++)
        {
            trajectory = (double *)&trajectories[i * trajectory_size];

            // Encode with the current synergies
            encode(&activities, trajectory, synergies, trajectory_length);

            // Decode with the current synergies
            decode(trajectory_reconstructed, &activities, synergies, trajectory_length);

            // Compute gradient in synergies
            add_gradient(gradient, synergies, &activities, trajectory, trajectory_reconstructed, trajectory_length);
        }

        // // Print magnitude of gradients
        // for (j = 0; j < synergies->n_synergies; j++)
        // {
        //     double total_gradient = 0.0;
        //     for (k = 0; k < synergies->synergy_length; k++)
        //     {
        //         for (l = 0; l < n_dim; l++)
        //         {
        //             total_gradient += gradient[(synergies->synergy_length * n_dim) * j + n_dim * k + l];
        //         }
        //     }

        //     printf("%d-th synergy gradient: %lf\n", j, total_gradient);
        // }
        // printf("\n");

        // Update synergies based on the gradient descent
        for (i = 0; i < synergies_size; i++)
        {
            synergies->synergies[i] -= -2.0 * lr * gradient[i];

            // Clip to non-negative values
            if (synergies->synergies[i] < 0.0)
            {
                synergies->synergies[i] = 0.0;
            }
        }

        // Normalize synergies
        normalize_synergies(synergies);

        if (print_progress)
        {
            printf("Extraction progress: %4.1lf%%\r", (double)(iter + 1) / (double)n_iter * 100.0);
        }
    }
    if (print_progress)
    {
        printf("\n");
    }

    finalize_activities(&activities);
    free(gradient);
    free(trajectory_reconstructed);

    return 0;
}

int encode(struct TVSynergyActivities *activities, const double *trajectory, const struct TimeVaryingSynergy *synergies, int trajectory_length)
{
    const double amplitude_th = 0.001;
    const int n_dim = synergies->n_dim;
    int *activity_index = calloc(synergies->n_synergies, sizeof(int));
    double *trajectory_copy = malloc(trajectory_length * n_dim * sizeof(double));
    int *synergy_available = calloc(activities->n_synergies * trajectory_length, sizeof(int)); // Whether the delay time of the synergy has been found
    double correlation;
    double max_correlation_value;
    double amplitude;
    int max_correlation_time;
    int max_correlation_synergy_idx;
    int idx;
    int idx_off_s;
    int idx_off_e;
    int i, j, k, l, n;

    // Copy a trajectory
    memcpy(trajectory_copy, trajectory, trajectory_length * n_dim * sizeof(double));

    // Initialize activities
    for (i = 0; i < activities->n_synergies; i++)
    {
        for (j = 0; j < activities->n_activities_max; j++)
        {
            activities->amplitudes[i * activities->n_activities_max + j] = 0.0;
            activities->delays[i * activities->n_activities_max + j] = 0;
        }
    }

    for (n = 0; n < activities->n_activities_max; n++)
    {
        max_correlation_value = 0.0;
        max_correlation_time = 0;
        max_correlation_synergy_idx = 0;

        // Compute correlations for all possible patterns
        for (i = 0; i < activities->n_synergies; i++)
        {
            for (j = 0; j < trajectory_length - synergies->synergy_length; j++)
            {
                if (synergy_available[trajectory_length * i + j] == 0)
                {
                    // Calculate correlations at time j with i-th synergy
                    correlation = 0.0;
                    for (k = 0; k < synergies->synergy_length; k++)
                    {
                        for (l = 0; l < n_dim; l++)
                        {
                            // correlations[i, j] = sum_{k, l}( trajectory_copy[j + k, l] * synergies->synergies[i, k, l] )
                            correlation += trajectory_copy[n_dim * (j + k) + l] * synergies->synergies[(synergies->synergy_length * n_dim) * i + n_dim * k + l];
                        }
                    }

                    if (correlation > max_correlation_value)
                    {
                        max_correlation_value = correlation;
                        max_correlation_time = j;
                        max_correlation_synergy_idx = i;
                    }
                }
            }
        }

        amplitude = max_correlation_value;

        if (amplitude < amplitude_th)
        {
            break;
        }

        idx = activities->n_activities_max * max_correlation_synergy_idx + activity_index[max_correlation_synergy_idx];
        activities->amplitudes[idx] = amplitude;
        activities->delays[idx] = max_correlation_time;
        activity_index[max_correlation_synergy_idx]++;

        // Compute residuals by subtracting the selected pattern
        for (k = 0; k < synergies->synergy_length; k++)
        {
            for (l = 0; l < n_dim; l++)
            {
                trajectory_copy[n_dim * (max_correlation_time + k) + l] -= amplitude * synergies->synergies[synergies->synergy_length * n_dim * max_correlation_synergy_idx + n_dim * k + l];
            }
        }

        // Remove the selected pattern and its surroundings
        idx_off_s = max_int(max_correlation_time - synergies->refractory_period, 0);
        idx_off_e = min_int(max_correlation_time + synergies->refractory_period, trajectory_length);
        for (k = idx_off_s; k < idx_off_e; k++)
        {
            synergy_available[trajectory_length * max_correlation_synergy_idx + k] = 1;
        }
    }

    // for (i = 0; i < activities->n_synergies; i++)
    // {
    //     for (j = 0; j < trajectory_length - synergies->synergy_length; j++)
    //     {
    //         printf("%d, ", synergy_available[trajectory_length * i + j]);
    //     }
    //     printf("\n");
    // }

    // Print synergy activity
    // for (j = 0; j < synergies->n_synergies; j++)
    // {
    //     for (k = 0; k < activities->n_activities_max; k++)
    //     {
    //         double amp = activities->amplitudes[activities->n_activities_max * j + k];
    //         int tau = activities->delays[activities->n_activities_max * j + k];

    //         if (amp == 0.0)
    //         {
    //             break;
    //         }
    //         printf("c_%d[%3d] = %lf\n", j, tau, amp);
    //     }
    // }
    // printf("\n");

    free(activity_index);
    free(trajectory_copy);
    free(synergy_available);

    return 0;
}

int decode(double *trajectory, const struct TVSynergyActivities *activities, const struct TimeVaryingSynergy *synergies, int trajectory_length)
{
    double amp;
    int tau;
    int i, j, k, l;
    int idx;

    // Initialize a trajectory
    for (i = 0; i < trajectory_length * synergies->n_dim; i++)
    {
        trajectory[i] = 0.0;
    }

    for (i = 0; i < activities->n_synergies; i++)
    {
        for (j = 0; j < activities->n_activities_max; j++)
        {
            amp = activities->amplitudes[i * activities->n_activities_max + j];
            tau = activities->delays[i * activities->n_activities_max + j];

            if (amp == 0.0)
            {
                break;
            }

            for (k = 0; k < synergies->synergy_length; k++)
            {
                if (tau + k >= trajectory_length)
                {
                    break;
                }

                for (l = 0; l < synergies->n_dim; l++)
                {
                    idx = synergies->synergy_length * synergies->n_dim * i + synergies->n_dim * k + l;
                    trajectory[synergies->n_dim * (tau + k) + l] += amp * synergies->synergies[idx];
                }
            }
        }
    }

    return 0;
}

void normalize_synergies(struct TimeVaryingSynergy *synergies)
{
    double squared_norm;
    double norm;
    int i, j, k;

    for (i = 0; i < synergies->n_synergies; i++)
    {
        // Calculate a squared norm of a synergy
        squared_norm = 0.0;
        for (j = 0; j < synergies->synergy_length; j++)
        {
            for (k = 0; k < synergies->n_dim; k++)
            {
                squared_norm += pow(synergies->synergies[synergies->synergy_length * synergies->n_dim * i + synergies->n_dim * j + k], 2);
            }
        }
        norm = sqrt(squared_norm);

        // Normalize the synergy
        for (j = 0; j < synergies->synergy_length; j++)
        {
            for (k = 0; k < synergies->n_dim; k++)
            {
                synergies->synergies[synergies->synergy_length * synergies->n_dim * i + synergies->n_dim * j + k] /= norm;
            }
        }
    }
}

void add_gradient(double *gradient, const struct TimeVaryingSynergy *synergies, const struct TVSynergyActivities *activities, const double *trajectory, const double *trajectory_reconstructed, int trajectory_length)
{
    double amp;
    int tau;
    int idx;
    int i, j, k, l;

    for (i = 0; i < synergies->n_synergies; i++)
    {
        for (j = 0; j < activities->n_activities_max; j++)
        {
            amp = activities->amplitudes[i * activities->n_activities_max + j];
            tau = activities->delays[i * activities->n_activities_max + j];

            // (amp == 0) means it reached the max number of synergy activity
            if (amp == 0.0)
            {
                break;
            }

            // printf("[%2d] %d-th activity: c_%d[%d] = %lf\n", i, j, i, tau, amp);

            for (k = 0; k < synergies->synergy_length; k++)
            {
                if (tau + k >= trajectory_length)
                {
                    break;
                }

                for (l = 0; l < synergies->n_dim; l++)
                {
                    idx = synergies->n_dim * (tau + k) + l;
                    gradient[(synergies->synergy_length * synergies->n_dim) * i + synergies->n_dim * k + l] += amp * (trajectory[idx] - trajectory_reconstructed[idx]);
                }
            }
        }
    }
}
