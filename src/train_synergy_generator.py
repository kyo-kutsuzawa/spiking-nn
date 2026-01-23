import argparse
import os
from dataclasses import dataclass, field
from logging import DEBUG, INFO, StreamHandler, getLogger
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm

from snn import SpikingNeuralNetwork


@dataclass
class Args:
    id: int = field(default_factory=int)
    """Synergy id to learn"""

    gain_in: float = field(default_factory=float)
    """Input gain"""

    gain_out: float = field(default_factory=float)
    """Output gain"""

    train_interval: int = field(default_factory=int)
    """Training interval [step]"""

    n_units: int = field(default_factory=int)
    """Number of units"""

    connection_ratio_x: float = field(default_factory=float)
    """Connection ratio between neurons"""

    connection_ratio_in: float = field(default_factory=float)
    """Connection ratio between input and neurons"""

    alpha: float = field(default_factory=float)
    """Reguralization factor"""

    G: float = field(default_factory=float)
    """Internal-feedback gain"""

    Q: float = field(default_factory=float)
    """Output-feedback gain"""

    bias: float = field(default_factory=float)
    """Bias for neurons"""

    t_end: float = field(default_factory=float)
    """Total simulation time [s]"""

    t_train_start: float = field(default_factory=float)
    """Time training starts [s]"""

    t_train_finish: float = field(default_factory=float)
    """Time training finishes [s]"""

    show_progress: bool = True
    """Whether to show a progress bar"""

    plot: bool = True
    """Whether to plot a figure"""


def train_synergy_model(args: Args) -> float:
    # Setup constants
    T: Final[float] = args.t_end
    t0: Final[float] = args.t_train_start
    t1: Final[float] = args.t_train_finish
    dt: Final[float] = 1.0 * 1e-3  # Integral time interval [s]
    nt: Final[int] = int(T / dt)  # Number of simulation loop
    train_interval: Final[int] = args.train_interval

    t_record: Final[float] = 0.0
    step: Final[int] = 1
    n_units_observed: Final[int] = 1

    # Load synergies
    filename_synergies: Final[str] = os.path.join(
        os.path.basename(__file__), "../dataset/synergies/synergy{}.csv".format(args.id)
    )
    dt_old: Final[float] = 0.05
    synergy = convert_synergies2(
        np.loadtxt(filename_synergies, delimiter=","), dt_old, dt
    )

    activation_pattern, trajectories = generate_random_activity2(synergy, T, dt)
    activation_pattern *= args.gain_in
    trajectories *= args.gain_out
    in_dim: Final[int] = 1
    out_dim: Final[int] = trajectories.shape[1]

    # Setup an SNN
    nn = SpikingNeuralNetwork(
        args.n_units,
        in_dim,
        out_dim,
        dt * 1e3,
        args.connection_ratio_x,
        args.connection_ratio_in,
        args.G,
        args.Q,
        args.alpha,
        args.bias,
    )
    nn.reset_state()

    # Initialize variables
    t = 0.0
    errors: list[float] = []
    Xest: list[npt.NDArray[np.float64]] = []
    Xteach: list[npt.NDArray[np.float64]] = []
    R: list[npt.NDArray[np.float64]] = []
    V: list[npt.NDArray[np.float64]] = []

    # Simulation loop
    for i in tqdm.tqdm(range(nt), disable=not args.show_progress):
        t = i * dt

        # Update the SNN
        nn.update(activation_pattern[i, :])
        xest = nn.x.copy()

        # Calculate the ground-truth
        x = trajectories[i]

        # Train the decoder
        if t0 < t < t1:
            if i % train_interval == 0:
                nn.train(x)

        # Compute error
        if t >= t1:
            e = float(np.sqrt(np.sum(((xest - x) / args.gain_out) ** 2)))
            errors.append(e)

        # Record the current states
        if t > t_record:
            if i % step == 0:
                Xest.append(xest)
                Xteach.append(x)
                R.append(nn.synapses.r[0:n_units_observed].copy())
                V.append(nn.neurons.v[0:n_units_observed].copy())

    # Compute the mean error after training finished
    mean_error: Final[float] = sum(errors) / len(errors)

    if args.plot:
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

        # Show the figure
        plt.show()

    return mean_error


def convert_synergies(
    synergy: npt.NDArray[np.float64], dt_old: float, dt_new: float
) -> npt.NDArray[np.float64]:

    length: Final[int] = synergy.shape[0]
    n_dim: Final[int] = synergy.shape[1]

    t_old = np.arange(length, dtype=np.float64) * dt_old
    t_new = np.linspace(0, t_old[-1], int(t_old[-1] / dt_new + 1), endpoint=True)

    synergy_new_list: list[npt.NDArray[np.float64]] = []
    for i in range(n_dim):
        v_new: npt.NDArray[np.float64] = np.interp(t_new, t_old, synergy[:, i])
        synergy_new_list.append(v_new.copy())

    synergy_new = np.stack(synergy_new_list, axis=1)

    return synergy_new


def convert_synergies2(
    synergy: npt.NDArray[np.float64], dt_old: float, dt_new: float
) -> npt.NDArray[np.float64]:

    length: Final[int] = synergy.shape[0]
    n_dim: Final[int] = synergy.shape[1] // 2

    t_old = np.arange(length, dtype=np.float64) * dt_old
    t_new = np.linspace(0, t_old[-1], int(t_old[-1] / dt_new + 1), endpoint=True)

    synergy_new_list: list[npt.NDArray[np.float64]] = []
    for i in range(n_dim):
        v_new = np.interp(t_new, t_old, synergy[:, i] - synergy[:, i + n_dim])
        synergy_new_list.append(v_new.copy())

    synergy_new = np.stack(synergy_new_list, axis=1)

    return synergy_new


def convert_activities(
    activity: npt.NDArray[np.float64], dt_old: float, dt_new: float
) -> npt.NDArray[np.float64]:

    length: Final[int] = activity.shape[0]
    n_dim: Final[int] = activity.shape[1]

    t_old = np.arange(length, dtype=np.float64) * dt_old
    t_new = np.linspace(0, t_old[-1], int(t_old[-1] / dt_new + 1), endpoint=True)

    activity_new = np.zeros((len(t_new), n_dim), dtype=np.float64)
    idx = 0
    for i, t in enumerate(t_new):
        activity_new[i, :] = activity[idx, :]

        if t > t_old[idx + 1]:
            idx += 1

    # activity_new_list: list[npt.NDArray[np.float64]] = []
    # for i in range(n_dim):
    #     v_new = np.interp(t_new, t_old, activity[:, i])
    #     activity_new_list.append(v_new.copy())

    # activity_new = np.stack(activity_new_list, axis=1)

    return activity_new


def pulse_to_persistence(
    activity: npt.NDArray[np.float64], synergy_length: int
) -> npt.NDArray[np.float64]:
    length: Final[int] = activity.shape[0]
    n_dim: Final[int] = activity.shape[1]

    activity_persistence = np.zeros_like(activity, dtype=np.float64)

    for i in range(synergy_length):
        activity_persistence[i:, :] += activity[: length - i, :]

    return activity_persistence


def generate_random_activity(
    synergy: npt.NDArray[np.float64], t_max: float, dt: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    synergy_length: Final[int] = synergy.shape[0]
    n_dim: Final[int] = synergy.shape[1]
    length: Final[int] = int(t_max / dt)
    n_activities: Final[int] = int(length / synergy_length * 0.5)
    refractory_period: Final[int] = synergy_length

    activation_pattern: npt.NDArray[np.float64] = np.zeros(
        (length, 1), dtype=np.float64
    )
    trajectories: npt.NDArray[np.float64] = np.zeros((length, n_dim), dtype=np.float64)

    synergy_available: npt.NDArray[np.bool] = np.full((length,), True, dtype=np.bool)
    synergy_available[length - synergy_length :] = False

    for i in range(n_activities):
        while True:
            tau = int(np.random.randint(0, length - synergy_length))
            amp = float(np.random.uniform(0.002, 0.02))
            if synergy_available[tau]:
                t0 = max(tau - refractory_period, 0)
                t1 = min(tau + refractory_period, length)
                synergy_available[t0:t1] = False
                break

        activation_pattern[tau : tau + synergy_length, 0] += amp
        trajectories[tau : tau + synergy_length, :] += synergy

    return activation_pattern, trajectories


def generate_random_activity2(
    synergy: npt.NDArray[np.float64], t_max: float, dt: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    synergy_length: Final[int] = synergy.shape[0]
    n_dim: Final[int] = synergy.shape[1]
    length: Final[int] = int(t_max / dt)

    # activation_pattern: npt.NDArray[np.float64] = np.zeros(
    #     (length, 1), dtype=np.float64
    # )

    # amp = 0.01
    # activation_pattern: npt.NDArray[np.float64] = np.full(
    #     (length, 1), amp, dtype=np.float64
    # )

    amp = 0.01
    activation_pattern: npt.NDArray[np.float64] = np.random.normal(
        0, amp, (length, 1)
    ).astype(np.float64)

    trajectories: npt.NDArray[np.float64] = np.zeros((length, n_dim), dtype=np.float64)

    for i in range(0, length, synergy_length):
        actual_length = min(i + synergy_length, length) - i
        trajectories[i : i + actual_length] = synergy[:actual_length]

    return activation_pattern, trajectories


def test_convert_synergies(args: Args) -> None:
    dt_old: Final[float] = 0.05
    dt: Final[float] = 1.0 * 1e-3  # Integral time interval [s]

    # Load a synergy
    filename: Final[str] = os.path.join(
        os.path.basename(__file__), "../dataset/synergies/synergy{}.csv".format(args.id)
    )
    synergy = np.loadtxt(filename, delimiter=",")
    synergy_new = convert_synergies(synergy, dt_old, dt)

    n_dim: Final[int] = synergy.shape[1]

    fig = plt.figure(constrained_layout=True)

    for i in range(n_dim):
        ax = fig.add_subplot(n_dim, 1, i + 1)
        ax.plot(
            np.arange(synergy.shape[0]) * dt_old,
            synergy[:, i],
            ls=":",
            color="C{}".format(i),
        )
        ax.plot(
            np.arange(synergy_new.shape[0]) * dt,
            synergy_new[:, i],
            lw=1,
            color="C{}".format(i),
        )

    plt.show()


def test_convert_synergies2(args: Args) -> None:
    dt_old: Final[float] = 0.05
    dt: Final[float] = 1.0 * 1e-3  # Integral time interval [s]

    # Load a synergy
    filename: Final[str] = os.path.join(
        os.path.basename(__file__), "../dataset/synergies/synergy{}.csv".format(args.id)
    )
    synergy = np.loadtxt(filename, delimiter=",")
    synergy_new = convert_synergies2(synergy, dt_old, dt)

    n_dim: Final[int] = synergy_new.shape[1]

    fig = plt.figure(constrained_layout=True)

    for i in range(n_dim):
        ax = fig.add_subplot(n_dim, 1, i + 1)
        ax.plot(
            np.arange(synergy.shape[0]) * dt_old,
            synergy[:, i],
            ls=":",
            color="C{}".format(i),
        )
        ax.plot(
            np.arange(synergy.shape[0]) * dt_old,
            -synergy[:, i + n_dim],
            ls="--",
            color="C{}".format(i),
        )
        ax.plot(
            np.arange(synergy_new.shape[0]) * dt,
            synergy_new[:, i],
            lw=1,
            color="C{}".format(i),
        )

    plt.show()


def test_convert_activities(args: Args) -> None:
    dt_old: Final[float] = 0.05
    dt: Final[float] = 1.0 * 1e-3  # Integral time interval [s]
    synergy_length: Final[int] = 20

    # Load synergy activity
    filename_activity: Final[str] = os.path.join(
        os.path.basename(__file__),
        "../dataset/train/data_converted{:02d}_activity.csv".format(args.id),
    )
    activity = np.loadtxt(filename_activity, delimiter=",")
    activity_persist = pulse_to_persistence(activity, synergy_length)
    activity_new = convert_activities(activity_persist, dt_old, dt)
    n_dim: Final[int] = activity.shape[1]

    fig = plt.figure(constrained_layout=True)

    for i in range(n_dim):
        ax = fig.add_subplot(n_dim, 1, i + 1)
        ax.plot(
            np.arange(activity.shape[0]) * dt_old,
            activity[:, i],
            ls=":",
            color="C{}".format(i),
        )
        ax.plot(
            np.arange(activity_new.shape[0]) * dt,
            activity_new[:, i],
            lw=1,
            color="C{}".format(i),
        )

    plt.show()


def test_generate_data(args: Args) -> None:
    T: Final[float] = 30.0  # Total simulation time [s]
    dt_old: Final[float] = 0.05
    dt: Final[float] = 1.0 * 1e-3  # Integral time interval [s]

    # Load synergies
    filename_synergies: Final[str] = os.path.join(
        os.path.basename(__file__), "../dataset/synergies/synergy{}.csv".format(args.id)
    )
    synergy = convert_synergies(
        np.loadtxt(filename_synergies, delimiter=","), dt_old, dt
    )

    activation_pattern, trajectories = generate_random_activity2(synergy, T, dt)
    activation_pattern *= args.gain_in
    trajectories *= args.gain_out
    length: Final[int] = activation_pattern.shape[0]

    fig = plt.figure(constrained_layout=True)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    tspace = np.linspace(0, T, length)
    ax1.plot(tspace, activation_pattern)
    ax2.plot(tspace, trajectories)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0, help="Synergy id to learn")
    parser.add_argument("--gain-in", type=float, default=0.0, help="Input gain")
    parser.add_argument("--gain-out", type=float, default=5.0, help="Output gain")
    parser.add_argument(
        "--train-interval", type=int, default=10, help="Training interval [step]"
    )
    parser.add_argument("--n-units", type=float, default=1000, help="Number of units")
    parser.add_argument(
        "--connection-ratio-x",
        type=float,
        default=0.01,
        help="Connection ratio between neurons",
    )
    parser.add_argument(
        "--connection-ratio-in",
        type=float,
        default=0.2,
        help="Connection ratio between input and neurons",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Reguralization factor"
    )
    parser.add_argument("--G", type=float, default=5e3, help="Internal-feedback gain")
    parser.add_argument("--Q", type=float, default=5e3, help="Output-feedback gain")
    parser.add_argument("--bias", type=float, default=1000.0, help="Bias for neurons")
    parser.add_argument(
        "--t-end", type=float, default=30.0, help="Total simulation time [s]"
    )
    parser.add_argument(
        "--t-train-start", type=float, default=0.0, help="Time training starts [s]"
    )
    parser.add_argument(
        "--t-train-finish", type=float, default=15.0, help="Time training finishes [s]"
    )
    __args = Args(**vars(parser.parse_args()))

    # Setup logger
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    # test_convert_synergies(__args)
    # test_convert_synergies2(__args)
    # test_convert_activities(__args)
    # test_generate_data(__args)
    train_synergy_model(__args)
    # train_activity_model(__args)
