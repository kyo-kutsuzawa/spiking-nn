import numpy as np
import numpy.typing as npt

from snn import SpikingNeuralNetwork


def example_SNN():
    import matplotlib.pyplot as plt
    import tqdm

    # Setup constants
    T = 30.0  # Total simulation time [s]
    t0 = 5.0
    t1 = 15.0
    dt = 1e-3  # Integral time interval [ms]
    nt = int(T / dt)  # Number of simulation loop
    step = 50
    train_interval = 10
    t_record = 0.0

    # Setup a neuron
    n_units = 1000
    connection_ratio = 0.01
    alpha = 1.0
    G = 5e3
    Q = 5e3
    bias = 1000.0
    nn = SpikingNeuralNetwork(
        n_units, 1, 1, dt * 1e3, connection_ratio, G, Q, alpha, bias
    )
    nn.reset_state()

    # Initialize variables
    a = 2 * np.pi * 5.0
    t = 0.0
    current = np.zeros((n_units,), dtype=np.float64)
    Xest: list[npt.NDArray[np.float64]] = []
    Xteach: list[npt.NDArray[np.float64]] = []
    R: list[npt.NDArray[np.float64]] = []
    V: list[npt.NDArray[np.float64]] = []

    # Simulation loop
    for i in tqdm.tqdm(range(nt)):
        # Update the SNN
        nn.update(current)
        xest = nn.x.copy()

        # Calculate the ground-truth
        _x = (
            0.1 * np.sin(a * t)
            + 0.5 * np.sin(0.7 * a * t + 1.0)
            + 0.3 * np.sin(0.44 * a * t + 2.1)
        )
        x = np.array([_x], dtype=np.float64)

        # Train the decoder
        if t0 < t < t1:
            if i % train_interval == 0:
                nn.train(x)

        # Record the current states
        t += dt

        if t > t_record:
            Xest.append(xest)
            Xteach.append(x)
            R.append(nn.synapses.r[0:5].copy())
            V.append(nn.neurons.v[0:5].copy())

    # Make a figure
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    # Plot results
    tspace = np.arange(nt)[::step] * dt
    tspace = np.linspace(t_record, T, len(Xest))
    ax1.plot(tspace, np.array(Xteach))
    ax1.plot(tspace, np.array(Xest))
    ax1.fill_between((t0, t1), -1.2, 1.2, color="black", alpha=0.3)
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
