#include <random>
#include <vector>
#include <Eigen/Core>

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class IzhikevichNeuron
{
public:
    Eigen::VectorXd v;
    Eigen::VectorXd u;
    IzhikevichNeuron();
    IzhikevichNeuron(int n_units, double dt);
    void reset_state();
    Eigen::VectorXd update(Eigen::Ref<const Eigen::VectorXd> input);
    int size();

private:
    int n_units;
    double dt;

    double C;       /**< @brief Membrane capacitance */
    double k;       /**< @brief Gain parameter of `vi` */
    double a;       /**< @brief Time scale parameter of `ui` */
    double b;       /**< @brief Sensitivity parameter of `ui` */
    double d;       /**< @brief After-spike reset parameter of `ui` */
    double vr;      /**< @brief Resting membrane potential */
    double vt;      /**< @brief Threshold voltage */
    double v_peak;  /**< @brief Peak voltage */
    double v_reset; /**< @brief Reset voltage */
};

class DoubleExponentialSynapticFilter
{
public:
    Eigen::VectorXd r;
    Eigen::VectorXd h;
    DoubleExponentialSynapticFilter();
    DoubleExponentialSynapticFilter(int n_units, double dt);
    void reset_state();
    void update(Eigen::Ref<const Eigen::VectorXd> spikes);
    int size();

private:
    int n_units;
    double dt;

    double tau_r;
    double tau_d;
};

class SpikingNeuralNetwork
{
public:
    IzhikevichNeuron neurons;
    DoubleExponentialSynapticFilter synapses;
    Eigen::VectorXd x;
    SpikingNeuralNetwork(int n_units, int in_size, int out_size, double dt, double connection_ratio, double G, double Q, double alpha, double bias);
    void reset_state();
    void update(Eigen::Ref<const Eigen::VectorXd> input);
    void train(Eigen::Ref<const Eigen::VectorXd> teaching_signal);
    int size();

private:
    int n_units;
    int in_size;
    int out_size;
    double dt;

    double p;               /**< @brief degree of sparsity in the network */
    double G;               /**< @brief scale of the static weight matrix */
    double Q;               /**< @brief scale of the feedback term */
    Eigen::VectorXd eta;    /**< @brief encoder that contributes to the tuning preferences of the neurons in the network */
    RowMatrixXd w0;         /**< @brief sparse and static weight matrix */
    RowMatrixXd phi;        /**< @brief decoder that is determined by RLS */
    Eigen::VectorXd i_bias; /**< @brief bias current */
    double l;               /**< @brief regularization parameter */
    RowMatrixXd P;          /**< @brief used for RLS */
    RowMatrixXd Gw0;
    RowMatrixXd Qeta;
};
