#ifndef _SNN_H_
#define _SNN_H_

struct IzhikevichNeurons
{
    double *v;
    double *u;

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

    double dt_c;
    double dt_a;
};

struct DoubleExponentialSynapticFilters
{
    double *r;
    double *h;

    int n_units;
    double dt;

    double tau_r;
    double tau_d;

    double gr_dt;
    double gd_dt;
    double g_rd;
};

struct SpikingNeuralNetwork
{
    // struct IzhikevichNeurons neurons;
    // struct DoubleExponentialSynapticFilters synapses;
    double *x;
    double *spikes;
    double *current;

    /* Network */
    int n_units;
    int in_size;
    int out_size;
    double dt;

    double p;       /**< @brief degree of sparsity in the network */
    double G;       /**< @brief scale of the static weight matrix */
    double Q;       /**< @brief scale of the feedback term */
    double *eta;    /**< @brief encoder that contributes to the tuning preferences of the neurons in the network */
    double *w0;     /**< @brief sparse and static weight matrix */
    double *phi;    /**< @brief decoder that is determined by RLS */
    double *i_bias; /**< @brief bias current */
    double l;       /**< @brief regularization parameter */
    double *P;      /**< @brief used for RLS */
    double *Gw0;
    double *Qeta;
    double *Pr;
    double *PrrP;

    /* Neurons */
    double *v;
    double *u;

    double C;       /**< @brief Membrane capacitance */
    double k;       /**< @brief Gain parameter of `vi` */
    double a;       /**< @brief Time scale parameter of `ui` */
    double b;       /**< @brief Sensitivity parameter of `ui` */
    double d;       /**< @brief After-spike reset parameter of `ui` */
    double vr;      /**< @brief Resting membrane potential */
    double vt;      /**< @brief Threshold voltage */
    double v_peak;  /**< @brief Peak voltage */
    double v_reset; /**< @brief Reset voltage */

    double dt_c;
    double dt_a;

    /* Synapses */
    double *r;
    double *h;

    double tau_r;
    double tau_d;

    double gr_dt;
    double gd_dt;
    double g_rd;

};

void initialize_snn(struct SpikingNeuralNetwork *snn, int n_units, int in_size, int out_size, double dt, double connection_ratio, double G, double Q, double alpha, double bias);
void reset_snn(struct SpikingNeuralNetwork *snn);
void update_snn(struct SpikingNeuralNetwork *snn, const double *input);
void train_snn(struct SpikingNeuralNetwork *snn, const double *teaching_signal);

void initialize_izhikevich_neurons(struct IzhikevichNeurons *neurons, int n_units, double dt);
void reset_izhikevich_neurons(struct IzhikevichNeurons *neurons);
void update_izhikevich_neurons(struct IzhikevichNeurons *neurons, double *spikes, const double *input);

void initialize_double_exponential_synaptic_filters(struct DoubleExponentialSynapticFilters *filters, int n_units, double dt);
void reset_double_exponential_synaptic_filters(struct DoubleExponentialSynapticFilters *filters);
void update_double_exponential_synaptic_filters(struct DoubleExponentialSynapticFilters *filters, const double *spikes);

#endif
