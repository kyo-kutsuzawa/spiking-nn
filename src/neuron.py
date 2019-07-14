import numpy as np


class IzhikevichNeuron:
    def __init__(self, n_units=1):
        self.C = 0.25  # Membrane capacitance
        self.k = 2.5  # Gain parameter of `vi`
        self.a = 10  # Time scale parameter of `ui`
        self.b = -2  # Sensitivity parameter of `ui`
        self.d = 200  # After-spike reset parameter of `ui`

        self.vr = -60  # Resting membrane potential
        self.vt = self.vr + 40 - self.b / self.k  # Threshold voltage
        self.v_peak = 30  # Peak voltage
        self.v_reset = -65  # Reset voltage

        self.dt = 1e-6  # Integral time interval [s]
        self.n_units = n_units

        # Initialize neuron states
        self.reset_state()

    def reset_state(self):
        self.ui = np.zeros((self.n_units,))
        self.vi = np.full((self.n_units,), self.vr, dtype=np.float64)

    def update(self, i):
        # Update the neuron states
        _vi = self.vi
        self.vi += self.dt / self.C * (self.k * (self.vi - self.vr) * (self.vi - self.vt) - self.ui + i)
        self.ui += self.dt * self.a * (self.b * (_vi - self.vr) - self.ui)

        # Reset the neuron states if necessary
        self.ui = np.where(self.vi >= self.v_peak, self.ui + self.d, self.ui)
        self.vi = np.where(self.vi >= self.v_peak, self.v_reset, self.vi)


class DoubleExponentialSynapticFilter:
    def __init__(self):
        self.tau_r = 2e-3  # Synaptic rise time
        self.tau_d = 2e-2  # Synaptic decay time

        self.dt = 1e-6  # Integral time interval [s]

        # Initialize synapse states
        self.reset_state()

    def reset_state(self):
        self.r = 0.0
        self.h = 0.0

    def update(self, spike):
        # Update the synapse states
        self.r = (1 - self.dt / self.tau_d) * self.r + self.h * self.dt
        self.h = (1 - self.dt / self.tau_r) * self.h + spike / (self.tau_r * self.tau_d)


def example_izhikevic():
    """An example of IzhikevichNeuron class.
    """
    import matplotlib.pyplot as plt

    # Setup a neuron
    n_units = 4
    neuron = IzhikevichNeuron(n_units=n_units)
    neuron.dt = 4e-5  # Integral time interval [s]

    # Setup constants
    T = 1.0  # Total simulation time [s]
    dt = neuron.dt
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
    ax_i.set_xlim((0, T))
    ax_u.set_xlim((0, T))
    ax_v.set_xlim((0, T))
    ax_i.set_ylabel("$i(t)$")
    ax_u.set_ylabel("$u(t)$")
    ax_v.set_ylabel("Mmembrane potential [mV]")
    ax_v.set_xlabel("Time [s]")

    # Show the figure
    plt.show()


def example_doubleESF():
    """An example of DoubleExponentialSynapticFilter class.
    """
    import matplotlib.pyplot as plt

    # Setup a neuron
    synapse = DoubleExponentialSynapticFilter()
    synapse.dt = 5e-5  # Integral time interval [s]

    # Setup constants
    T = 0.1  # Total simulation time [s]
    dt = synapse.dt
    nt = int(T / dt)  # Number of simulation loop

    # Initialize variables
    H = []
    R = []
    t = 0.0
    t_spike = 0.01  # [s]

    # Simulation loop
    for i in range(nt):
        # Update a spike
        if i == int(t_spike / dt):
            spike = 1
        else:
            spike = 0

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
    ax_h.set_xlim((0, T))
    ax_r.set_xlim((0, T))
    ax_h.set_ylabel("$h_r(t)$")
    ax_r.set_ylabel("$r(t)$")
    ax_r.set_xlabel("Time [s]")

    # Show the figure
    plt.show()


if __name__ == "__main__":
    example_izhikevic()
