#include "synergy.hpp"

TimeVaryingSynergy::TimeVaryingSynergy(int n_synergies, int synergy_length, int n_dims)
{
    int i, j, k;
    this->n_synergies = n_synergies;
    this->synergy_length = synergy_length;
    this->n_dims = n_dims;
    this->refractory_period = synergy_length / 2;

    this->synergies = std::vector<std::vector<std::vector<double>>>(this->n_synergies, std::vector<std::vector<double>>(this->synergy_length, std::vector<double>(this->n_dims, 0.0)));
}

void TimeVaryingSynergy::extract(const std::vector<std::vector<std::vector<double>>> &trajectories, int n_iter, double lr)
{
    const int amplitude_th = 0.001;
    int n_trajectories = trajectories.size();
    int iter;
    int i, j, k;
    int trajectory_length;
    std::vector<std::vector<std::vector<double>>> gradient(this->n_synergies, std::vector<std::vector<double>>(this->synergy_length, std::vector<double>(this->n_dims)));
    std::vector<std::vector<std::vector<double>>> amplitudes(n_trajectories, std::vector<std::vector<double>>());
    std::vector<std::vector<std::vector<int>>> delays(n_trajectories, std::vector<std::vector<int>>());
    std::vector<std::vector<double>> trajectory_reconstructed;

    for (iter = 0; iter < n_iter; iter++)
    {
        // Reset the gradient
        for (i = 0; i < this->n_synergies; i++)
        {
            for (j = 0; j < this->synergy_length; j++)
            {
                std::fill(gradient[i][j].begin(), gradient[i][j].end(), 0.0);
            }
        }

        // Compute gradient for each trajectory
        for (i = 0; i < n_trajectories; i++)
        {
            // Get the trajectory length
            trajectory_length = trajectories[i].size();

            trajectory_reconstructed = std::vector<std::vector<double>>(trajectory_length, std::vector<double>(this->n_dims, 0.0));

            // Encode with the current synergies
            this->encode(trajectories[i], amplitudes[i], delays[i]);

            // Decode with the current synergies
            this->decode(amplitudes[i], delays[i], trajectory_reconstructed);

            
        }

        // Update synergies based on the gradient descent
    }

    // n_data = len(dataset)
    // grad = np.zeros_like(synergies)

    // for n in range(n_data):
    //     data = dataset[n]

    //     # Compute reconstruction data
    //     data_est = np.zeros_like(data)
    //     for k in range(synergies.shape[0]):
    //         for ts, c in zip(delays[n][k], amplitude[n][k]):
    //             data_est[ts:ts+synergies.shape[1], :] += c * synergies[k, :, :]

    //     # Compute the gradient
    //     deviation = data - data_est
    //     for k in range(synergies.shape[0]):
    //         for ts, c in zip(delays[n][k], amplitude[n][k]):
    //             #data_est[ts:ts+synergies.shape[1], :] += c * synergies[k, :, :]
    //             grad[k, :, :] += deviation[ts:ts+synergies.shape[1], :] * c

    // # Compute the gradient
    // grad = grad * -2

    // # Update the amplitude
    // synergies = synergies - mu * grad
    // synergies = np.clip(synergies, 0.0, None)  # Limit to non-negative values

    // for k in range(synergies.shape[0]):
    //     norm = np.sqrt(np.sum(np.square(synergies[k])))
    //     synergies[k] = synergies[k] / float(norm)

    // return synergies

    // for i in range(args.n_iter):
    //     delays, amplitude = timevarying.match_synergies(movements, synergies, args.n_synergies_use, refractory_period)

    //     r2 = timevarying.compute_R2(movements, synergies, amplitude, delays)
    //     print("Iter {:4d}: R2 = {}".format(i, r2))

    //     # Save synergies
    //     np.save(os.path.join(args.out, "synergy.npy"), synergies)

    //     synergies = timevarying.update_synergies(movements, synergies, amplitude, delays, args.lr)
}

void TimeVaryingSynergy::encode(const std::vector<std::vector<double>> &trajectory, std::vector<std::vector<double>> &amplitudes, std::vector<std::vector<int>> &delays)
{
    amplitudes = std::vector<std::vector<double>>(this->n_synergies, std::vector<double>());
    delays = std::vector<std::vector<int>>(this->n_synergies, std::vector<int>());
}

void TimeVaryingSynergy::decode(const std::vector<std::vector<double>> &amplitudes, const std::vector<std::vector<int>> &delays, std::vector<std::vector<double>> &trajectory)
{
    int trajectory_length = trajectory.size();
    int n_activities;
    double amp;
    int tau;
    int i, j, k, l;

    for (i = 0; i < this->n_synergies; i++)
    {
        n_activities = amplitudes[i].size();

        for (j = 0; j < n_activities; j++)
        {
            amp = amplitudes[i][j];
            tau = delays[i][j];

            for (k = 0; k < this->synergy_length; k++)
            {
                for (l = 0; l < this->n_dims; l++)
                {
                    trajectory[tau + k][l] += amp * this->synergies[i][k][l];
                }
            }
        }
    }
}

// inline double &TimeVaryingSynergy::synergies_at(int i, int j, int k)
// {
//     return this->synergies[i * (this->n_synergies * this->synergy_length) + j * this->synergy_length + k];
// }
