#ifndef _SYNEGY_H_
#define _SYNEGY_H_

struct TimeVaryingSynergy
{
    double *synergies;
    int n_synergies;
    int synergy_length;
    int n_dim;
    int refractory_period;
};

struct TVSynergyActivities
{
    double *amplitudes;
    int *delays;
    int n_synergies;
    int n_activities_max;
};

int extract(struct TimeVaryingSynergy *synergies, int n_synergies, int synergy_length, int n_dim, int refractory_period, const double *trajectories, int n_data, int trajectory_length, int n_iter, double lr, int n_activities_max);
int encode(struct TVSynergyActivities *activities, const double *trajectory, const struct TimeVaryingSynergy *synergies, int trajectory_length, int n_dim);
int decode(double *trajectory, const struct TVSynergyActivities *activities, const struct TimeVaryingSynergy *synergies, int trajectory_length);

#endif
