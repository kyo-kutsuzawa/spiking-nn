from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm

from snn import SpikingNeuralNetwork


def example_SNN() -> None:
    # Setup constants
    T: Final[float] = 30.0  # Total simulation time [s]
    t0: Final[float] = 0.0
    t1: Final[float] = 15.0
    dt: Final[float] = 1.0 * 1e-3  # Integral time interval [s]
    nt: Final[int] = int(T / dt)  # Number of simulation loop
    train_interval: Final[int] = 10

    t_record: Final[float] = 0.0
    step: Final[int] = 1
    n_units_observed: Final[int] = 1

    # Setup an SNN
    n_units: Final[int] = 1000
    out_dim: Final[int] = 2
    connection_ratio_x: Final[float] = 0.01
    connection_ratio_in: Final[float] = 0.2
    alpha: Final[float] = 1.0
    G: Final[float] = 5e3
    Q: Final[float] = 5e3
    bias: Final[float] = 1000.0
    nn = SpikingNeuralNetwork(
        n_units,
        1,
        out_dim,
        dt * 1e3,
        connection_ratio_x,
        connection_ratio_in,
        G,
        Q,
        alpha,
        bias,
    )
    nn.reset_state()

    # Initialize variables
    a: Final[float] = 2 * np.pi * 5.0
    t = 0.0
    current = np.zeros((1,), dtype=np.float64)
    Xest: list[npt.NDArray[np.float64]] = []
    Xteach: list[npt.NDArray[np.float64]] = []
    R: list[npt.NDArray[np.float64]] = []
    V: list[npt.NDArray[np.float64]] = []

    # Simulation loop
    for i in tqdm.tqdm(range(nt)):
        t = i * dt
        current = np.random.normal(0, 100, (1,)).astype(np.float64).reshape(-1)

        # Update the SNN
        nn.update(current)
        xest = nn.x.copy()

        # Calculate the ground-truth
        _x = (
            0.1 * np.sin(a * t)
            + 0.5 * np.sin(0.7 * a * t + 1.0)
            + 0.3 * np.sin(0.44 * a * t + 2.1)
        )
        _y = 0.5 * (
            0.1 * np.cos(a * t)
            + 0.5 * np.cos(0.7 * a * t + 1.0)
            + 0.3 * np.cos(0.44 * a * t + 2.1)
        )
        x = np.array([_x, _y], dtype=np.float64)

        # Train the decoder
        if t0 < t < t1:
            if i % train_interval == 0:
                nn.train(x)

        # Record the current states
        if t > t_record:
            if i % step == 0:
                Xest.append(xest)
                Xteach.append(x)
                R.append(nn.synapses.r[0:n_units_observed].copy())
                V.append(nn.neurons.v[0:n_units_observed].copy())

    # Make a figure
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    # Plot results
    tspace = np.linspace(t_record, T, len(Xest))
    for i in range(out_dim):
        ax1.plot(tspace, np.array(Xteach)[:, i], color="C{}".format(i), ls=":")
        ax1.plot(tspace, np.array(Xest)[:, i], color="C{}".format(i))
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

    # # Plot on the x-y plane
    # fig2 = plt.figure(figsize=(12, 4), constrained_layout=True)
    # ax2_1 = fig2.add_subplot(1, 1, 1)
    # t_plot = int(20.0 / dt)
    # ax2_1.plot(np.array(Xteach)[t_plot:, 0], np.array(Xteach)[t_plot:, 1], ls=":")
    # ax2_1.plot(np.array(Xest)[t_plot:, 0], np.array(Xest)[t_plot:, 1])

    # Show the figure
    plt.show()


if __name__ == "__main__":
    example_SNN()
