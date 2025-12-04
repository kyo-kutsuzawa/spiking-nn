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

int initialize_synergies(struct TimeVaryingSynergy *synergies, int n_synergies, int synergy_length, int n_dim, int refractory_period);
int finalize_synergies(struct TimeVaryingSynergy *synergies);
int initialize_activities(struct TVSynergyActivities *activities, int n_synergies, int n_activities_max);
int finalize_activities(struct TVSynergyActivities *activities);

int extract(struct TimeVaryingSynergy *synergies, const double *trajectories, int n_data, int trajectory_length, int n_dim, int n_iter, double lr, int n_activities_max);
int encode(struct TVSynergyActivities *activities, const double *trajectory, const struct TimeVaryingSynergy *synergies, int trajectory_length);
int decode(double *trajectory, const struct TVSynergyActivities *activities, const struct TimeVaryingSynergy *synergies, int trajectory_length);

#endif
