import numpy as np


class IzhikevichNeuron:
    def __init__(self):
        self.C = 250  # Membrane capacitance
        self.k = 2.5
        self.a = 0.01
        self.b = -2
        self.d = 200

        self.vr = -60  # Resting membrane potential
        self.vt = self.vr + 40 - self.b / self.k  # Threshold
        self.v_peak = 30
        self.v_reset = -65

        self.dt = 0.001

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

    # Make figures
    fig = plt.figure(figsize=(6, 6))
    ax_i = fig.add_subplot(3, 1, 1)
    ax_v = fig.add_subplot(3, 1, 2)
    ax_u = fig.add_subplot(3, 1, 3)

    # Plot results
    tspace = np.arange(nt) * dt * 1e-3
    ax_i.plot(tspace, np.array(I))
    ax_u.plot(tspace, np.array(U))
    ax_v.plot(tspace, np.array(V))

    # Set labels
    ax_i.set_xlim((0, T*1e-3))
    ax_u.set_xlim((0, T*1e-3))
    ax_v.set_xlim((0, T*1e-3))
    ax_i.set_ylabel("$i(t)$")
    ax_u.set_ylabel("$u(t)$")
    ax_v.set_ylabel("Mmembrane potential [mV]")
    ax_v.set_xlabel("Time [s]")

    plt.show()


if __name__ == "__main__":
    example_izhikevic()
