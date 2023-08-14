import numpy as np
import neuron


class STDPLayer:
    def __init__(self, in_dim, out_dim, dt):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dt = dt

        self.amp_p = 0.03
        self.amp_d = 0.03 * 0.85
        self.tau_p = 16.8
        self.tau_d = 33.7

        self.amp_p = 0.03
        self.amp_d = 0.03 * 0.85

        self.synapses = neuron.SingleExponentialSynapticFilter(self.in_dim)
        self.synapses.dt = self.dt

        self.weights = np.full((self.in_dim, self.out_dim), 1e2)

        self.reset_state()

    def update(self, spike_in):
        # Compute continuous input
        self.synapses.update(spike_in)
        u = np.dot(self.synapses.r, self.weights)

        return u

    def record_spikes(self, spike_in, spike_out):
        # Update the input firing trace
        self.trace_in -= self.trace_in * self.dt / self.tau_p
        self.trace_in += spike_in
        self.spike_in = spike_in

        # Update the output firing trace
        self.trace_out -= self.trace_out * self.dt / self.tau_d
        self.trace_out += spike_out
        self.spike_out = spike_out

    def train(self):
        dw_ltp = self.amp_p * np.einsum("i, j->ij", self.trace_in, self.spike_out)
        dw_ltd = self.amp_d * np.einsum("i, j->ij", self.spike_in, self.trace_out)
        self.weights += dw_ltp - dw_ltd

        self.weights = np.clip(self.weights, 0.0, 1e3)

    def reset_state(self):
        self.synapses.reset_state()
        self.trace_in = np.zeros((self.in_dim,))
        self.trace_out = np.zeros((self.out_dim,))
        self.spike_in = np.zeros((self.in_dim,))


def example_stdp():
    import matplotlib.pyplot as plt

    n_steps = 10000
    spike_interval = 1500
    n_units = 10
    dt = 0.1  # [ms]

    layer = STDPLayer(n_units, 1, dt)

    S = []
    W = []
    T = []

    for i in range(n_steps):
        spike_in = np.zeros((n_units,))
        spike_out = np.zeros((1,))

        if i % spike_interval == 0:
            spike_out[0] = 1.0

        for j in range(n_units):
            if (i + j * 150 + 50) % spike_interval == 0:
                spike_in[j] = 1.0

        layer.update(spike_in)
        layer.record_spikes(spike_in, spike_out)
        layer.train()

        S.append(np.concatenate((spike_in, spike_out)))
        W.append(layer.weights.copy())
        T.append(np.concatenate((layer.trace_in, layer.trace_out)))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)

    times = np.arange(n_steps)
    W = np.array(W)
    S = np.array(S)
    T = np.array(T)
    for j in range(n_units):
        ax1.plot(times, S[:, j])
        ax2.plot(times, W[:, j])
        ax3.plot(times, T[:, j])

    ax1.plot(times, S[:, -1], color="black")
    ax3.plot(times, T[:, -1], color="black")

    ax1.set_ylabel("Spike")
    ax2.set_ylabel("Weight")
    ax3.set_ylabel("Firing trace")
    ax1.set_xlim(0, n_steps)
    ax2.set_xlim(0, n_steps)
    ax3.set_xlim(0, n_steps)

    plt.show()


if __name__ == "__main__":
    example_stdp()
