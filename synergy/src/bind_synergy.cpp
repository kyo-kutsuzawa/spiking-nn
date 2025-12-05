extern "C"
{
#include "synergy.h"
}
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

class _TimeVaryingSynergy
{
public:
    struct TimeVaryingSynergy data;
    _TimeVaryingSynergy();
    _TimeVaryingSynergy(int n_synergies, int synergy_length, int n_dims, int refractory_period);
    std::vector<std::vector<std::vector<double>>> get_synergies();
};

_TimeVaryingSynergy _extract(const std::vector<std::vector<std::vector<double>>> &trajectories, int n_synergies, int synergy_length, int refractory_period, int n_activities_max, int n_iter, double lr, bool print_progress);
pybind11::tuple _encode(const std::vector<std::vector<double>> &trajectory, const _TimeVaryingSynergy &synergies, int n_activities_max);
std::vector<std::vector<double>> _decode(const std::vector<std::vector<double>> &amplitudes, const std::vector<std::vector<int>> &delays, const _TimeVaryingSynergy &synergies, int trajectory_length);

PYBIND11_MODULE(synergy, m)
{
    m.doc() = "Synergy";

    pybind11::class_<_TimeVaryingSynergy>(m, "TimeVaryingSynergy")
        .def(pybind11::init<int, int, int, int>())
        .def("get_synergies", &_TimeVaryingSynergy::get_synergies);

    m.def("extract", &_extract, "Extract synergies");
    m.def("encode", &_encode, "Encode");
    m.def("decode", &_decode, "Decode");
}

_TimeVaryingSynergy::_TimeVaryingSynergy()
{
}

_TimeVaryingSynergy::_TimeVaryingSynergy(int n_synergies, int synergy_length, int n_dim, int refractory_period)
{
    initialize_synergies(&this->data, n_synergies, synergy_length, n_dim, refractory_period);
}

std::vector<std::vector<std::vector<double>>> _TimeVaryingSynergy::get_synergies()
{
    std::vector<std::vector<std::vector<double>>> _synergies(this->data.n_synergies, std::vector<std::vector<double>>(this->data.synergy_length, std::vector<double>(this->data.n_dim)));

    for (int i = 0; i < _synergies.size(); i++)
    {
        for (int j = 0; j < _synergies[i].size(); j++)
        {
            for (int k = 0; k < _synergies[i][j].size(); k++)
            {
                _synergies[i][j][k] = this->data.synergies[(this->data.synergy_length * this->data.n_dim) * i + this->data.n_dim * j + k];
            }
        }
    }

    return _synergies;
}

_TimeVaryingSynergy _extract(const std::vector<std::vector<std::vector<double>>> &trajectories, int n_synergies, int synergy_length, int refractory_period, int n_activities_max, int n_iter, double lr, bool print_progress)
{
    const size_t n_data = trajectories.size();
    const size_t trajectory_length = trajectories[0].size();
    const size_t n_dim = trajectories[0][0].size();
    const size_t trajectories_size = n_data * trajectory_length * n_dim;
    double *trajectories_array = (double *)malloc(trajectories_size * sizeof(double));
    _TimeVaryingSynergy synergies(n_synergies, synergy_length, (int)n_dim, refractory_period);

    for (size_t i = 0; i < n_data; i++)
    {
        for (size_t j = 0; j < trajectory_length; j++)
        {
            for (size_t k = 0; k < n_dim; k++)
            {
                trajectories_array[(trajectory_length * n_dim) * i + n_dim * j + k] = trajectories[i][j][k];
            }
        }
    }

    extract(&(synergies.data), trajectories_array, (int)n_data, (int)trajectory_length, (int)n_dim, n_iter, lr, n_activities_max, (int)print_progress);

    free(trajectories_array);

    return synergies;
}

pybind11::tuple _encode(const std::vector<std::vector<double>> &trajectory, const _TimeVaryingSynergy &synergies, int n_activities_max)
{
    const size_t trajectory_length = trajectory.size();
    const size_t n_dim = trajectory[0].size();
    const size_t trajectory_size = trajectory_length * n_dim;
    double *trajectory_array = (double *)malloc(trajectory_size * sizeof(double));
    struct TVSynergyActivities activities;
    std::vector<std::vector<double>> amplitudes(synergies.data.n_synergies, std::vector<double>(n_activities_max, 0.0));
    std::vector<std::vector<int>> delays(synergies.data.n_synergies, std::vector<int>(n_activities_max, 0));
    pybind11::tuple activities_tuple;

    initialize_activities(&activities, synergies.data.n_synergies, n_activities_max);

    for (size_t i = 0; i < trajectory_length; i++)
    {
        for (size_t j = 0; j < n_dim; j++)
        {
            trajectory_array[n_dim * i + j] = trajectory[i][j];
        }
    }

    encode(&activities, trajectory_array, &(synergies.data), (int)trajectory_length);

    for (size_t i = 0; i < activities.n_synergies; i++)
    {
        for (size_t j = 0; j < activities.n_activities_max; j++)
        {
            amplitudes[i][j] = activities.amplitudes[activities.n_activities_max * i + j];
            delays[i][j] = activities.delays[activities.n_activities_max * i + j];
        }
    }
    activities_tuple = pybind11::make_tuple(amplitudes, delays);

    finalize_activities(&activities);

    return activities_tuple;
}

std::vector<std::vector<double>> _decode(const std::vector<std::vector<double>> &amplitudes, const std::vector<std::vector<int>> &delays, const _TimeVaryingSynergy &synergies, int trajectory_length)
{
    const size_t n_activities_max=amplitudes[0].size();;
    const int n_dim = synergies.data.n_dim;
    struct TVSynergyActivities activities;
    double *trajectory_array = (double *)calloc(trajectory_length * n_dim, sizeof(double));
    std::vector<std::vector<double>> trajectory(trajectory_length, std::vector<double>(n_dim, 0.0));
    int idx;

    // Initialize synergy activity
    initialize_activities(&activities, synergies.data.n_synergies, (int)n_activities_max);

    // Copy synergy activity data
    for (int i = 0; i < activities.n_synergies; i++)
    {
        for (int j = 0; j < activities.n_activities_max; j++)
        {
            idx = activities.n_activities_max * i + j;
            activities.amplitudes[idx] = amplitudes[i][j];
            activities.delays[idx] = delays[i][j];
        }
    }

    decode(trajectory_array, &activities, &synergies.data, trajectory_length);

    // Copy reconstructed trajectory
    for (int i = 0; i < trajectory_length; i++)
    {
        for (int j = 0; j < n_dim; j++)
        {
            trajectory[i][j] = trajectory_array[n_dim * i + j];
        }
    }

    free(trajectory_array);
    finalize_activities(&activities);

    return trajectory;
}
