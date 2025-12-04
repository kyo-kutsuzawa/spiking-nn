#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "synergy.h"

double rand_uniform01(void)
{
    return (double)rand() / RAND_MAX;
}

int extract(struct TimeVaryingSynergy *synergies, int n_synergies, int synergy_length, int n_dim, int refractory_period, const double *trajectories, int n_data, int trajectory_length, int n_iter, double lr, int n_activities_max)
{
    struct TVSynergyActivities activities;
    double *gradient;
    double *trajectory;
    double *trajectory_reconstructed;
    double amp;
    int tau;
    double synergy_norm;
    double synergy_norm_squared;
    int synergies_size;
    int trajectory_size;
    int i, j, k, l, m;
    int idx;
    int iter;

    srand((unsigned int)time(NULL));

    synergies_size = n_synergies * synergy_length * n_dim;
    trajectory_size = trajectory_length * n_dim;

    // Initialize synergies
    synergies->synergies = (double *)malloc(synergies_size * sizeof(double));
    synergies->n_synergies = n_synergies;
    synergies->synergy_length = synergy_length;
    synergies->n_dim = n_dim;
    synergies->refractory_period = refractory_period;

    // Initialize synergy activities
    activities.amplitudes = (double *)calloc(n_synergies * n_activities_max, sizeof(double));
    activities.delays = (int *)calloc(n_synergies * n_activities_max, sizeof(int));
    activities.n_synergies = n_synergies;
    activities.n_activities_max = n_activities_max;

    // Initialize array variables
    gradient = (double *)malloc(synergies_size * sizeof(double));
    trajectory_reconstructed = (double *)calloc(trajectory_size, sizeof(double));

    // Return -1 if memory allocation failed
    if ((synergies->synergies == NULL) || (gradient == NULL) || (trajectory_reconstructed == NULL) || (activities.amplitudes == NULL) || (activities.delays == NULL))
    {
        return -1;
    }

    for (i = 0; i < synergies_size; i++)
    {
        synergies->synergies[i] = rand_uniform01();
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
            encode(&activities, trajectory, synergies, trajectory_length, n_dim);

            // Decode with the current synergies
            decode(trajectory_reconstructed, &activities, synergies, trajectory_length);

            // Calculate gradient
            for (j = 0; j < n_synergies; j++)
            {
                for (k = 0; k < n_activities_max; k++)
                {
                    amp = activities.amplitudes[j * n_activities_max + k];
                    tau = activities.delays[j * n_activities_max + k];

                    // (amp == 0) means it reached the max number of synergy activity
                    if (amp == 0.0)
                    {
                        break;
                    }

                    for (l = 0; l < synergy_length; l++)
                    {
                        if (tau + l >= trajectory_length)
                        {
                            break;
                        }

                        for (m = 0; m < n_dim; m++)
                        {
                            idx = n_dim * (tau + l) + m;
                            gradient[(synergy_length * n_dim) * j + n_dim * l + m] += amp * (trajectory[idx] - trajectory_reconstructed[idx]);
                        }
                    }
                }
            }
        }

        // Update synergies based on the gradient descent
        for (i = 0; i < synergies_size; i++)
        {
            synergies->synergies[i] -= -2.0 * lr * gradient[i];
        }

        // Normalize synergies
        for (i = 0; i < n_synergies; i++)
        {
            // Calculate a norm of a synergy
            synergy_norm_squared = 0.0;
            for (j = 0; j < synergy_length; j++)
            {
                for (k = 0; k < n_dim; k++)
                {
                    synergy_norm_squared += pow(synergies->synergies[(synergy_length * n_dim) * i + n_dim * j + k], 2);
                }
            }
            synergy_norm = sqrt(synergy_norm_squared);

            // Normalize the synergy
            for (j = 0; j < synergy_length; j++)
            {
                for (k = 0; k < n_dim; k++)
                {
                    synergies->synergies[(synergy_length * n_dim) * i + n_dim * j + k] /= synergy_norm;
                }
            }
        }
    }

    return 0;
}

int encode(struct TVSynergyActivities *activities, const double *trajectory, const struct TimeVaryingSynergy *synergies, int trajectory_length, int n_dim)
{
    // // Initialize synergy activities
    // activities.amplitudes = (double *)calloc(n_synergies * n_activities_max, sizeof(double));
    // activities.delays = (int *)calloc(n_synergies * n_activities_max, sizeof(int));
    // activities.n_synergies = n_synergies;
    // activities.n_activities_max = n_activities_max;

    return 0;
}

int decode(double *trajectory, const struct TVSynergyActivities *activities, const struct TimeVaryingSynergy *synergies, int trajectory_length)
{
    double amp;
    int tau;
    int i, j, k, l;
    int idx;

    // Initialize a trajectory
    if (trajectory == NULL)
    {
        trajectory = (double *)calloc(trajectory_length * synergies->n_dim, sizeof(double));
    }
    else
    {
        for (i = 0; i < trajectory_length * synergies->n_dim; i++)
        {
            trajectory[i] = 0.0;
        }
    }

    for (i = 0; i < synergies->n_synergies; i++)
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

// void TimeVaryingSynergy::extract(const std::vector<std::vector<std::vector<double>>> &trajectories, int n_iter, double lr)
// {
//     const int amplitude_th = 0.001;
//     int n_trajectories = trajectories.size();
//     int iter;
//     int i, j, k;
//     int trajectory_length;
//     std::vector<std::vector<std::vector<double>>> gradient(this->n_synergies, std::vector<std::vector<double>>(this->synergy_length, std::vector<double>(this->n_dims)));
//     std::vector<std::vector<std::vector<double>>> amplitudes(n_trajectories, std::vector<std::vector<double>>());
//     std::vector<std::vector<std::vector<int>>> delays(n_trajectories, std::vector<std::vector<int>>());
//     std::vector<std::vector<double>> trajectory_reconstructed;

//     for (iter = 0; iter < n_iter; iter++)
//     {
//         // Reset the gradient
//         for (i = 0; i < this->n_synergies; i++)
//         {
//             for (j = 0; j < this->synergy_length; j++)
//             {
//                 std::fill(gradient[i][j].begin(), gradient[i][j].end(), 0.0);
//             }
//         }

//         // Compute gradient for each trajectory
//         for (i = 0; i < n_trajectories; i++)
//         {
//             // Get the trajectory length
//             trajectory_length = trajectories[i].size();

//             trajectory_reconstructed = std::vector<std::vector<double>>(trajectory_length, std::vector<double>(this->n_dims, 0.0));

//             // Encode with the current synergies
//             this->encode(trajectories[i], amplitudes[i], delays[i]);

//             // Decode with the current synergies
//             this->decode(amplitudes[i], delays[i], trajectory_reconstructed);
//         }

//         // Update synergies based on the gradient descent
//     }

//     // n_data = len(dataset)
//     // grad = np.zeros_like(synergies)

//     // for n in range(n_data):
//     //     data = dataset[n]

//     //     # Compute reconstruction data
//     //     data_est = np.zeros_like(data)
//     //     for k in range(synergies.shape[0]):
//     //         for ts, c in zip(delays[n][k], amplitude[n][k]):
//     //             data_est[ts:ts+synergies.shape[1], :] += c * synergies[k, :, :]

//     //     # Compute the gradient
//     //     deviation = data - data_est
//     //     for k in range(synergies.shape[0]):
//     //         for ts, c in zip(delays[n][k], amplitude[n][k]):
//     //             #data_est[ts:ts+synergies.shape[1], :] += c * synergies[k, :, :]
//     //             grad[k, :, :] += deviation[ts:ts+synergies.shape[1], :] * c

//     // # Compute the gradient
//     // grad = grad * -2

//     // # Update the amplitude
//     // synergies = synergies - mu * grad
//     // synergies = np.clip(synergies, 0.0, None)  # Limit to non-negative values

//     // for k in range(synergies.shape[0]):
//     //     norm = np.sqrt(np.sum(np.square(synergies[k])))
//     //     synergies[k] = synergies[k] / float(norm)

//     // return synergies

//     // for i in range(args.n_iter):
//     //     delays, amplitude = timevarying.match_synergies(movements, synergies, args.n_synergies_use, refractory_period)

//     //     r2 = timevarying.compute_R2(movements, synergies, amplitude, delays)
//     //     print("Iter {:4d}: R2 = {}".format(i, r2))

//     //     # Save synergies
//     //     np.save(os.path.join(args.out, "synergy.npy"), synergies)

//     //     synergies = timevarying.update_synergies(movements, synergies, amplitude, delays, args.lr)
// }

// void TimeVaryingSynergy::encode(const std::vector<std::vector<double>> &trajectory, std::vector<std::vector<double>> &amplitudes, std::vector<std::vector<int>> &delays)
// {
//     amplitudes = std::vector<std::vector<double>>(this->n_synergies, std::vector<double>());
//     delays = std::vector<std::vector<int>>(this->n_synergies, std::vector<int>());
// }

// void TimeVaryingSynergy::decode(const std::vector<std::vector<double>> &amplitudes, const std::vector<std::vector<int>> &delays, std::vector<std::vector<double>> &trajectory)
// {
//     int trajectory_length = trajectory.size();
//     int n_activities;
//     double amp;
//     int tau;
//     int i, j, k, l;

//     for (i = 0; i < this->n_synergies; i++)
//     {
//         n_activities = amplitudes[i].size();

//         for (j = 0; j < n_activities; j++)
//         {
//             amp = amplitudes[i][j];
//             tau = delays[i][j];

//             for (k = 0; k < this->synergy_length; k++)
//             {
//                 for (l = 0; l < this->n_dims; l++)
//                 {
//                     trajectory[tau + k][l] += amp * this->synergies[i][k][l];
//                 }
//             }
//         }
//     }
// }

// // inline double &TimeVaryingSynergy::synergies_at(int i, int j, int k)
// // {
// //     return this->synergies[i * (this->n_synergies * this->synergy_length) + j * this->synergy_length + k];
// // }
