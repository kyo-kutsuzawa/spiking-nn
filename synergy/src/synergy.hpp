#include <vector>

class TimeVaryingSynergy
{
public:
    std::vector<std::vector<std::vector<double>>> synergies;
    TimeVaryingSynergy(int n_synergies, int synergy_length, int n_dims);
    void extract(const std::vector<std::vector<std::vector<double>>> &trajectories, int n_iter, double lr);
    void encode(const std::vector<std::vector<double>> &trajectory, std::vector<std::vector<double>> &amplitudes, std::vector<std::vector<int>> &delays);
    void decode(const std::vector<std::vector<double>> &amplitudes, const std::vector<std::vector<int>> &delays, std::vector<std::vector<double>> &trajectory);

private:
    int n_synergies;
    int synergy_length;
    int n_dims;
    int refractory_period;
};
