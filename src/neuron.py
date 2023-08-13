import numpy as np


class LIFNeuron:
    """leaky integrate-and-fire neuron model.

    A neuron has two status: ui and vi.
    """

    def __init__(self, n_units=1):
        """Initialization.

        Args:
            n_units (int): Number of neurons.
        """
        self.tau_m = 10.0  # Membrane time constant [ms]
        self.v_threshold = -40  # Threshold voltage
        self.v_peak = 30  # Peak voltage
        self.v_reset = -65  # Reset voltage
        self.refractory_period = 2.0  # Refractory time [ms]

        self.dt = 1e-3  # Integral time interval [ms]
        self.n_units = n_units

        # Initialize neuron states
        self.reset_state()

    def reset_state(self):
        """Reset the neuron states to zero."""
        self.vi = np.random.uniform(self.v_reset, self.v_threshold, size=(self.n_units))
        self.t_spike = np.zeros((self.n_units,))
        self.cnt = 0

    def update(self, i):
        """Update the neuron states.

        Args:
            i (numpy.ndarray): Inputs to the neurons.
        """
        # Reset the neuron states if necessary
        self.vi = np.where(self.vi >= self.v_peak, self.v_reset, self.vi)

        # Update the neuron states
        dv = self.dt / self.tau_m * (-self.vi + i)
        dv = np.where(
            self.cnt * self.dt >= self.t_spike + self.refractory_period, dv, 0
        )
        self.vi += dv

        # Detect spiking
        spike = np.where(self.vi >= self.v_threshold, 1, 0)
        self.vi = np.where(spike, self.v_peak, self.vi)
        self.t_spike = np.where(spike, self.cnt * self.dt, self.t_spike)

        self.cnt += 1

        return spike


class IzhikevichNeuron:
    """Neuron model based on the Izhikevich model.

    A neuron has two status: ui and vi.
    """

    def __init__(self, n_units=1):
        """Initialization.

        Args:
            n_units (int): Number of neurons.
        """
        self.C = 250  # Membrane capacitance
        self.k = 2.5  # Gain parameter of `vi`
        self.a = 0.01  # Time scale parameter of `ui`
        self.b = -2  # Sensitivity parameter of `ui`
        self.d = 200  # After-spike reset parameter of `ui`

        self.vr = -60  # Resting membrane potential
        self.vt = self.vr + 40 - self.b / self.k  # Threshold voltage
        self.v_peak = 30  # Peak voltage
        self.v_reset = -65  # Reset voltage

        self.dt = 1e-3  # Integral time interval [ms]
        self.n_units = n_units

        # Initialize neuron states
        self.reset_state()

    def reset_state(self):
        """Reset the neuron states to zero."""
        self.ui = np.zeros((self.n_units,))
        self.vi = self.vr + (self.v_peak - self.vr) * np.random.uniform(
            0, 1, size=(self.n_units)
        )

    def update(self, i):
        """Update the neuron states.

        Args:
            i (numpy.ndarray): Inputs to the neurons.
        """
        # Update the neuron states
        _vi = self.vi
        self.vi += (
            self.dt
            / self.C
            * (self.k * (self.vi - self.vr) * (self.vi - self.vt) - self.ui + i)
        )
        self.ui += self.dt * self.a * (self.b * (_vi - self.vr) - self.ui)

        # Detect spiking
        spike = np.where(self.vi >= self.v_peak, 1, 0)

        # Reset the neuron states if necessary
        self.ui = np.where(self.vi >= self.v_peak, self.ui + self.d, self.ui)
        self.vi = np.where(self.vi >= self.v_peak, self.v_reset, self.vi)

        return spike


class SingleExponentialSynapticFilter:
    """Synapse model based on single exponential synaptic filter."""

    def __init__(self, n_units=1):
        """Initialization.

        Args:
            n_units (int): Number of synapses.
        """
        self.tau_d = 20  # Synaptic decay time [ms]

        self.dt = 1e-3  # Integral time interval [ms]
        self.n_units = n_units

        # Initialize synapse states
        self.reset_state()

    def reset_state(self):
        """Reset the synapse states to zero."""
        self.r = np.zeros((self.n_units,))

    def update(self, spike):
        """Update the synapse states

        Args:
            spike (numpy.ndarray): Input spikes
        """
        self.r = (1 - self.dt / self.tau_d) * self.r + spike / self.tau_d


class DoubleExponentialSynapticFilter:
    """Synapse model based on double exponential synaptic filter."""

    def __init__(self, n_units=1):
        """Initialization.

        Args:
            n_units (int): Number of synapses.
        """
        self.tau_r = 2  # Synaptic rise time
        self.tau_d = 20  # Synaptic decay time

        self.dt = 1e-3  # Integral time interval [ms]
        self.n_units = n_units

        # Initialize synapse states
        self.reset_state()

    def reset_state(self):
        """Reset the synapse states to zero."""
        self.r = np.zeros((self.n_units,))
        self.h = np.zeros((self.n_units,))

    def update(self, spike):
        """Update the synapse states

        Args:
            spike (numpy.ndarray): Input spikes
        """
        self.r = (1 - self.dt / self.tau_d) * self.r + self.h * self.dt
        self.h = (1 - self.dt / self.tau_r) * self.h + spike / (self.tau_r * self.tau_d)


def example_lif():
    """An example of LIFNeuron class.

    Random input current based on a Gaussian distribution is used.
    """
    import matplotlib.pyplot as plt

    # Setup a neuron
    n_units = 4
    neuron = LIFNeuron(n_units=n_units)
    neuron.dt = 4e-2  # Integral time interval [ms]

    # Setup constants
    T = 1.0  # Total simulation time [s]
    dt = neuron.dt * 1e-3
    nt = int(T / dt)  # Number of simulation loop

    # Initialize variables
    I = []
    V = []
    bias = -40  # pA

    # Simulation loop
    for _ in range(nt):
        # Update the neuron
        i = np.random.normal(0, 10, size=(n_units,))  # pA
        neuron.update(i + bias)

        # Record the current states
        I.append(i)
        V.append(neuron.vi)

    # Make a figure
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax_i = fig.add_subplot(2, 1, 1)
    ax_v = fig.add_subplot(2, 1, 2)

    # Plot results
    tspace = np.arange(nt) * dt
    ax_i.plot(tspace, np.array(I))
    ax_v.plot(tspace, np.array(V))

    # Setup the figure
    fig.suptitle("Simulation of LIFNeuron")
    # ax_i.set_title("Input current")
    # ax_u.set_title("Membrane potential")
    ax_i.set_xlim((0, T))
    ax_v.set_xlim((0, T))
    ax_i.set_ylabel("Input current $i(t)$")
    ax_v.set_ylabel("Mmembrane potential $v(i)$ [mV]")
    ax_v.set_xlabel("Time [s]")

    # Show the figure
    plt.show()


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
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
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
    # ax_i.set_title("Input current")
    # ax_u.set_title("Internal State")
    # ax_u.set_title("Membrane potential")
    ax_i.set_xlim((0, T))
    ax_u.set_xlim((0, T))
    ax_v.set_xlim((0, T))
    ax_i.set_ylabel("Input current $i(t)$")
    ax_u.set_ylabel("$u(t)$")
    ax_v.set_ylabel("Mmembrane potential $v(i)$ [mV]")
    ax_v.set_xlabel("Time [s]")

    # Show the figure
    plt.show()


def example_singleESF():
    """An example of SingleExponentialSynapticFilter class.

    Spikes at random timing are used.
    """
    import matplotlib.pyplot as plt

    # Setup a neuron
    n_units = 4
    synapse = SingleExponentialSynapticFilter(n_units=n_units)
    synapse.dt = 5e-2  # Integral time interval [ms]

    # Setup constants
    T = 0.1  # Total simulation time [s]
    dt = synapse.dt * 1e-3
    nt = int(T / dt)  # Number of simulation loop

    # Initialize variables
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
        R.append(synapse.r)

        t += dt

    # Make a figure
    fig = plt.figure(constrained_layout=True)
    ax_r = fig.add_subplot(1, 1, 1)

    # Plot results
    tspace = np.arange(nt) * dt
    ax_r.plot(tspace, np.array(R))

    # Setup the figure
    fig.suptitle("Simulation of SingleExponentialSynapticFilter")
    ax_r.set_xlim((0, T))
    ax_r.set_ylabel("$r(t)$")
    ax_r.set_xlabel("Time [s]")

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
    fig = plt.figure(constrained_layout=True)
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


if __name__ == "__main__":
    example_lif()
    example_izhikevic()
    example_singleESF()
    example_doubleESF()
