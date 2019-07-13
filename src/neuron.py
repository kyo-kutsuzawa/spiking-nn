import numpy as np


class IzhikevichNeuron:
    def __init__(self):
        self.C = 250  # Membrane capacitance
        self.k = 2.5  # Gain parameter of `vi`
        self.a = 0.01  # Time scale parameter of `ui`
        self.b = -2  # Sensitivity parameter of `ui`
        self.d = 200  # After-spike reset parameter of `ui`

        self.vr = -60  # Resting membrane potential
        self.vt = self.vr + 40 - self.b / self.k  # Threshold voltage
        self.v_peak = 30  # Peak voltage
        self.v_reset = -65  # Reset voltage

        self.dt = 0.001  # Integral time interval [ms]

        # Initialize neuron states
        self.reset_state()

    def reset_state(self):
        self.ui = 0.0
        self.vi = self.vr

    def update(self, i):
        # Update the neuron states
        _vi = self.vi
        self.vi += self.dt / self.C * (self.k * (self.vi - self.vr) * (self.vi - self.vt) - self.ui + i)
        self.ui += self.dt * self.a * (self.b * (_vi - self.vr) - self.ui)

        # Reset the neuron states if necessary
        if self.vi >= self.v_peak:
            self.ui += self.d
            self.vi = self.v_reset


def example_izhikevic():
    """An example of IzhikevichNeuron class.
    """
    import matplotlib.pyplot as plt

    # Setup a neuron
    neuron = IzhikevichNeuron()
    neuron.dt = 0.04  # Integral time interval [ms]

    # Setup constants
    T = 1000  # Total simulation time [ms]
    dt = neuron.dt
    nt = int(T / dt)  # Number of simulation loop

    # Initialize variables
    I = []
    U = []
    V = []

    # Simulation loop
    for _ in range(nt):
        # Update the neuron
        bias = 1000
        i = np.random.normal(100, 300)
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
    tspace = np.arange(nt) * dt * 1e-3
    ax_i.plot(tspace, np.array(I))
    ax_u.plot(tspace, np.array(U))
    ax_v.plot(tspace, np.array(V))

    # Setup the figure
    ax_i.set_xlim((0, T*1e-3))
    ax_u.set_xlim((0, T*1e-3))
    ax_v.set_xlim((0, T*1e-3))
    ax_i.set_ylabel("$i(t)$")
    ax_u.set_ylabel("$u(t)$")
    ax_v.set_ylabel("Mmembrane potential [mV]")
    ax_v.set_xlabel("Time [s]")

    # Show the figure
    plt.show()


if __name__ == "__main__":
    example_izhikevic()
