import numpy as np
from neuron import IzhikevichNeuron, DoubleExponentialSynapticFilter


class SpikingNN:
    """Spiking neural network (SNN) with FORCE learning.

    This SNN consists of
    1) neurons with the Izhikevich model and
    2) synapses with the double exponential synaptic filter.
    """

    def __init__(self, n_units, in_size, out_size):
        """Initialization.

        Args:
            n_units (int): Number of neurons.
        """
        self.neurons = IzhikevichNeuron(n_units=n_units)
        self.synapses = DoubleExponentialSynapticFilter(n_units=n_units)

        self.p = 0.1  # degree of sparsity in the network
        self.G = 5e3  # scale of the static weight matrix
        self.Q = 5e3  # scale of the feedback term
        self.eta = np.random.uniform(
            -1.0, 1.0, size=(n_units, out_size)
        )  # encoder that contributes to the tuning preferences of the neurons in the network
        self.w0 = np.random.normal(
            0, 1 / (np.sqrt(n_units) * self.p), size=(n_units, n_units)
        )  # sparse and static weight matrix
        self.phi = np.zeros((n_units, out_size))  # decoder that is determined by RLS.
        self.i_bias = 1000  # bias current

        self.mask = np.where(
            np.random.uniform(0, 1, size=(n_units, n_units)) < self.p, 1, 0
        )
        self.w0 *= self.mask

        l = 2.0  # regularization parameter
        self.P = np.identity(n_units) / l  # used for RLS

        self.n_units = n_units

        self.Gw0 = self.G * self.w0.T
        self.Qeta = self.Q * self.eta.T

    def reset_state(self):
        self.neurons.reset_state()
        self.synapses.reset_state()
        self.x = np.dot(self.synapses.r, self.phi)

    def update(self):
        # Calculate input currents
        s = np.dot(self.synapses.r, self.Gw0) + np.dot(self.x, self.Qeta)
        i = s + self.i_bias

        # Update the states of neurons and synapses
        spikes = self.neurons.update(i)
        self.synapses.update(spikes)

        # Calculate the output, x
        self.x = np.dot(self.synapses.r, self.phi)
        return self.x

    def train(self, teaching_signal):
        err = self.x - teaching_signal

        # Update P
        Pr = self.P.dot(self.synapses.r.T).reshape((-1, 1))
        rPr = self.synapses.r.dot(Pr)
        c = 1.0 / (1.0 + rPr)
        self.P -= Pr.dot(Pr.T) * c

        # Update phi
        self.phi -= err * Pr


def example_SNN():
    """An example of SpikingNN class.

    Spikes at random timing are used.
    """
    import matplotlib.pyplot as plt
    import tqdm

    # Setup constants
    T = 15.0  # Total simulation time [s]
    dt = 1e-3
    nt = int(T / dt)  # Number of simulation loop
    step = 50
    train_interval = 10
    t_record = 0.0

    # Setup a neuron
    n_units = 500
    nn = SpikingNN(n_units=n_units, in_size=1, out_size=1)
    nn.neurons.dt = dt * 1e3  # Integral time interval [ms]
    nn.synapses.dt = dt * 1e3  # Integral time interval [ms]
    nn.reset_state()

    # Initialize variables
    a = 2 * np.pi * 5.0
    t = 0.0
    Xest = []
    Xteach = []
    R = []
    V = []

    # Simulation loop
    for i in tqdm.tqdm(range(nt)):
        # Update the SNN
        xest = nn.update()

        # Calculate the ground-truth
        x = np.sin(a * t)

        # Train the decoder
        if 1.0 < t < 10.0:
            if i % train_interval == 0:
                nn.train(x)

        # Record the current states
        t += dt

        if t > t_record:
            Xest.append(xest)
            Xteach.append(x)
            R.append(nn.synapses.r[0:5])
            V.append(nn.neurons.vi[0:5])

    # Make a figure
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    # Plot results
    tspace = np.arange(nt)[::step] * dt
    tspace = np.linspace(t_record, T, len(Xest))
    ax1.plot(tspace, np.array(Xest))
    ax1.plot(tspace, np.array(Xteach))
    ax2.plot(tspace, np.array(R))
    ax3.plot(tspace, np.array(V))

    # Setup the figure
    fig.suptitle("Simulation of SpikingNN")
    ax1.set_xlim((t_record, T))
    ax2.set_xlim((t_record, T))
    ax3.set_xlim((t_record, T))
    ax1.set_ylabel("$x(t)$")
    ax2.set_ylabel("$r(t)$")
    ax3.set_ylabel("$v(t)$")
    ax1.set_xlabel("Time [s]")

    # Show the figure
    plt.show()


if __name__ == "__main__":
    example_SNN()
