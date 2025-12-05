#include <random>
#include <vector>
#include <Eigen/Core>
#include <Eigen/SparseCore>

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class IzhikevichNeuron
{
public:
    Eigen::VectorXd v; /**< @brief [mV] */
    Eigen::VectorXd u; /**< @brief [pA] */
    IzhikevichNeuron();
    IzhikevichNeuron(int n_units, double dt);
    void reset_state();
    void update(Eigen::Ref<Eigen::VectorXd> spikes, const Eigen::Ref<const Eigen::VectorXd> input);
    int size();

private:
    int n_units; /**< @brief Number of neurons */
    double dt;   /**< @brief Computation interval [ms] */

    Eigen::VectorXd v_pre;

    double C;       /**< @brief Membrane capacitance [uF] */
    double k;       /**< @brief Gain parameter of `vi` [nS] */
    double a;       /**< @brief Time scale parameter of `ui` [(ms)^{-1}] */
    double b;       /**< @brief Sensitivity parameter of `ui` [nS] */
    double d;       /**< @brief After-spike reset parameter of `ui` [pA] */
    double vr;      /**< @brief Resting membrane potential [mV] */
    double vt;      /**< @brief Threshold voltage [mV] */
    double v_peak;  /**< @brief Peak voltage [mV] */
    double v_reset; /**< @brief Reset voltage [mV] */

    double dt_C; /**< @brief dt / C [MOhm] */
    double dt_a; /**< @brief dt * a [-] */
    Eigen::VectorXd vr_vec;
    Eigen::VectorXd vt_vec;
};

class DoubleExponentialSynapticFilter
{
public:
    Eigen::VectorXd r;
    Eigen::VectorXd h;
    DoubleExponentialSynapticFilter();
    DoubleExponentialSynapticFilter(int n_units, double dt);
    void reset_state();
    void update(const Eigen::Ref<const Eigen::VectorXd> spikes);
    int size();

private:
    int n_units;
    double dt;

    double tau_r /**< @brief [ms] */;
    double tau_d /**< @brief [ms] */;
};

class SpikingNeuralNetwork
{
public:
    IzhikevichNeuron neurons;
    DoubleExponentialSynapticFilter synapses;
    Eigen::VectorXd x;
    SpikingNeuralNetwork(int n_units, int in_size, int out_size, double dt, double connection_ratio_x, double connection_ratio_in, double G, double Q, double alpha, double bias);
    void reset_state();
    void update(const Eigen::Ref<const Eigen::VectorXd> input);
    void train(const Eigen::Ref<const Eigen::VectorXd> teaching_signal);
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
    Eigen::VectorXd i_bias; /**< @brief bias current [pA] */
    double alpha;           /**< @brief regularization parameter */
    RowMatrixXd P;          /**< @brief used for RLS */
    RowMatrixXd Gw0;
    RowMatrixXd Qeta;
    RowMatrixXd Win;
    Eigen::VectorXd errors;
    Eigen::VectorXd Pr;
    RowMatrixXd PrrP;
    Eigen::VectorXd current;
    Eigen::VectorXd spikes;

    Eigen::SparseMatrix<double> w0_sp;
    Eigen::SparseMatrix<double> Gw0_sp;
    Eigen::SparseMatrix<double> win_sp;
};
