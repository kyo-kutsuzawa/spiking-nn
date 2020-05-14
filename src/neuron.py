import numpy as np


class IzhikevichNeuron:
    """Neuron model based on the Izhikevich model.

    A neuron has two status: ui and vi.
    """
    def __init__(self, n_units=1):
        """Initialization.

        Args:
            n_units (int): Number of neurons.
        """
        self.C = 250   # Membrane capacitance
        self.k = 2.5   # Gain parameter of `vi`
        self.a = 0.01  # Time scale parameter of `ui`
        self.b = -2    # Sensitivity parameter of `ui`
        self.d = 200   # After-spike reset parameter of `ui`

        self.vr = -60  # Resting membrane potential
        self.vt = self.vr + 40 - self.b / self.k  # Threshold voltage
        self.v_peak = 30  # Peak voltage
        self.v_reset = -65  # Reset voltage

        self.dt = 1e-6  # Integral time interval [s]
        self.n_units = n_units

        # Initialize neuron states
        self.reset_state()

    def reset_state(self):
        """Reset the neuron states to zero.
        """
        self.ui = np.zeros((self.n_units,))
        self.vi = self.vr + (self.v_peak - self.vr) * np.random.uniform(0, 1, size=(self.n_units))

    def update(self, i):
        """Update the neuron states.

        Args:
            i (numpy.ndarray): Inputs to the neurons.
        """
        # Update the neuron states
        _vi = self.vi
        self.vi += self.dt / self.C * (self.k * (self.vi - self.vr) * (self.vi - self.vt) - self.ui + i)
        self.ui += self.dt * self.a * (self.b * (_vi - self.vr) - self.ui)

        # Detect spiking
        spike = np.where(self.vi >= self.v_peak, 1, 0)

        # Reset the neuron states if necessary
        self.ui = np.where(self.vi >= self.v_peak, self.ui + self.d, self.ui)
        self.vi = np.where(self.vi >= self.v_peak, self.v_reset, self.vi)

        return spike


class DoubleExponentialSynapticFilter:
    """Synapse model based on double exponential synaptic filter.
    """
    def __init__(self, n_units=1):
        """Initialization.

        Args:
            n_units (int): Number of synapses.
        """
        self.tau_r =  2  # Synaptic rise time
        self.tau_d = 20  # Synaptic decay time

        self.dt = 1e-6  # Integral time interval [s]
        self.n_units = n_units

        # Initialize synapse states
        self.reset_state()

    def reset_state(self):
        """Reset the synapse states to zero.
        """
        self.r = np.zeros((self.n_units,))
        self.h = np.zeros((self.n_units,))

    def update(self, spike):
        """Update the synapse states

        Args:
            spike (numpy.ndarray): Input spikes
        """
        self.r = (1 - self.dt / self.tau_d) * self.r + self.h * self.dt
        self.h = (1 - self.dt / self.tau_r) * self.h + spike / (self.tau_r * self.tau_d)


class SpikingNN:
    """Spiking neural network (SNN).

    An SNN consists of
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
        self.eta = np.random.uniform(-1.0, 1.0, size=(n_units, out_size))  # encoder that contributes to the tuning preferences of the neurons in the network
        self.w0 = np.random.normal(0, 1/(np.sqrt(n_units)*self.p), size=(n_units, n_units))  # sparse and static weight matrix
        self.phi = np.zeros((n_units, out_size))  # decoder that is determined by RLS.
        self.i_bias = 1000  # bias current

        self.mask = np.where(np.random.uniform(0, 1, size=(n_units, n_units)) < self.p, 1, 0)
        self.w0 *= self.mask

        l = 1 / 0.001  # regularization parameter
        l = 2.0
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
        Pr = self.P.dot(self.synapses.r.T).reshape((-1,1))
        rPr = self.synapses.r.dot(Pr)
        c = 1.0 / (1.0 + rPr)
        self.P -= Pr.dot(Pr.T) * c

        # Update phi
        self.phi -= err * Pr


def example_izhikevic():
    """An example of IzhikevichNeuron class.

    Random input current based on a Gaussian distribution is used.
    """
    import matplotlib.pyplot as plt

    # Setup a neuron
    n_units = 4
    neuron = IzhikevichNeuron(n_units=n_units)
    neuron.dt = 4e-2  # Integral time interval [ms]

    # Setup constants
    T = 1.0  # Total simulation time [s]
    dt = neuron.dt * 1e-3
    nt = int(T / dt)  # Number of simulation loop

    # Initialize variables
    I = []
    U = []
    V = []
    bias = np.random.uniform(1000, 1200, (n_units,))

    # Simulation loop
    for _ in range(nt):
        # Update the neuron
        i = np.random.normal(100, 300, size=(n_units,))
        neuron.update(i + bias)

        # Record the current states
        I.append(i)
        U.append(neuron.ui)
        V.append(neuron.vi)

    # Make a figure
    fig = plt.figure(figsize=(6, 6))
    ax_i = fig.add_subplot(3, 1, 1)
    ax_u = fig.add_subplot(3, 1, 2)
    ax_v = fig.add_subplot(3, 1, 3)

    # Plot results
    tspace = np.arange(nt) * dt
    ax_i.plot(tspace, np.array(I))
    ax_u.plot(tspace, np.array(U))
    ax_v.plot(tspace, np.array(V))

    # Setup the figure
    fig.suptitle("Simulation of IzhikevichNeuron")
    #ax_i.set_title("Input current")
    #ax_u.set_title("Internal State")
    #ax_u.set_title("Membrane potential")
    ax_i.set_xlim((0, T))
    ax_u.set_xlim((0, T))
    ax_v.set_xlim((0, T))
    ax_i.set_ylabel("Input current $i(t)$")
    ax_u.set_ylabel("$u(t)$")
    ax_v.set_ylabel("Mmembrane potential $v(i)$ [mV]")
    ax_v.set_xlabel("Time [s]")

    # Show the figure
    plt.show()


def example_doubleESF():
    """An example of DoubleExponentialSynapticFilter class.

    Spikes at random timing are used.
    """
    import matplotlib.pyplot as plt

    # Setup a neuron
    n_units = 4
    synapse = DoubleExponentialSynapticFilter(n_units=n_units)
    synapse.dt = 5e-2  # Integral time interval [ms]

    # Setup constants
    T = 0.1  # Total simulation time [s]
    dt = synapse.dt * 1e-3
    nt = int(T / dt)  # Number of simulation loop

    # Initialize variables
    H = []
    R = []
    t = 0.0
    t_spike = np.random.uniform(0, 0.02, size=(n_units,))  # Timing of spikes [s]

    # Simulation loop
    for i in range(nt):
        # Update a spike
        spike = np.zeros((n_units,))
        for j in range(n_units):
            if i == int(t_spike[j] / dt):
                spike[j] = 1

        # Update the synapse
        synapse.update(spike)

        # Record the current states
        H.append(synapse.h)
        R.append(synapse.r)

        t += dt

    # Make a figure
    fig = plt.figure()
    ax_h = fig.add_subplot(2, 1, 1)
    ax_r = fig.add_subplot(2, 1, 2)

    # Plot results
    tspace = np.arange(nt) * dt
    ax_h.plot(tspace, np.array(H))
    ax_r.plot(tspace, np.array(R))

    # Setup the figure
    fig.suptitle("Simulation of DoubleExponentialSynapticFilter")
    ax_h.set_xlim((0, T))
    ax_r.set_xlim((0, T))
    ax_h.set_ylabel("$h_r(t)$")
    ax_r.set_ylabel("$r(t)$")
    ax_r.set_xlabel("Time [s]")

    # Show the figure
    plt.show()


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
    t_record = 14.0

    # Setup a neuron
    n_units = 2000
    nn = SpikingNN(n_units=n_units, in_size=1, out_size=1)
    nn.neurons.dt  = dt*1e3  # Integral time interval [ms]
    nn.synapses.dt = dt*1e3  # Integral time interval [ms]
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
        if 5.0 < t < 10.0:
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
    fig = plt.figure()
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
    #example_izhikevic()
    #example_doubleESF()
    example_SNN()
