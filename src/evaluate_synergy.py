import argparse
import glob
import os
import random
from dataclasses import dataclass, field
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import synergy


@dataclass
class Args:
    n_synergies: int = field(default_factory=int)
    synergy_length: int = field(default_factory=int)
    n_iter: int = field(default_factory=int)
    lr: float = field(default_factory=float)


def evaluate(args: Args) -> None:
    # Define constants
    n_synergies: Final[int] = args.n_synergies
    synergy_length: Final[int] = args.synergy_length
    n_dof: Final[int] = 2
    # refractory_period: Final[int] = int(synergy_length / 2)
    refractory_period: Final[int] = synergy_length
    n_activities_max: Final[int] = 70
    n_iter: Final[int] = args.n_iter
    lr: Final[float] = args.lr
    n_show: Final[int] = 5

    # Load a dataset
    dataset: list[npt.NDArray[np.float64]] = []
    datasets_dir = os.path.join(
        os.path.basename(__file__), "../dataset/data_realsense_train/*.csv"
    )
    filelist = glob.glob(datasets_dir)
    for filename in filelist:
        data = np.loadtxt(filename, delimiter=",")
        dataset.append(data)

    # Create result folders
    synergy_dir: Final[str] = os.path.join(
        os.path.basename(__file__), "../dataset/synergies"
    )
    os.makedirs(synergy_dir, exist_ok=True)
    activity_dir: Final[str] = os.path.join(
        os.path.basename(__file__), "../dataset/train"
    )
    os.makedirs(activity_dir, exist_ok=True)

    # Preprosessing the dataset
    trajectories = convert_dataset(dataset)
    n_data: Final[int] = len(trajectories)
    trajectory_length: Final[int] = len(trajectories[0])

    # Extract synergies
    tvsynergies = synergy.extract(
        trajectories,
        n_synergies,
        synergy_length,
        refractory_period,
        n_activities_max,
        n_iter,
        lr,
        True,
    )

    synergies = np.array(tvsynergies.get_synergies(), dtype=np.float64)

    # Save synergies
    for i in range(n_synergies):
        filename_synergy = os.path.join(synergy_dir, "synergy{}.csv".format(i))
        np.savetxt(filename_synergy, synergies[i], delimiter=",")

    # Save activities
    for i in range(n_data):
        amplitudes, delays = synergy.encode(
            trajectories[i], tvsynergies, n_activities_max
        )
        activity = convert_activity(amplitudes, delays, trajectory_length)
        filename_activity = os.path.join(
            activity_dir,
            os.path.splitext(os.path.basename(filelist[i]))[0] + "_activity.csv",
        )
        np.savetxt(filename_activity, activity, delimiter=",")

    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs_master = GridSpec(nrows=1, ncols=3, figure=fig, width_ratios=[2, 2, 1])

    # Plot reconstruction data
    gs_2d = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 0])
    gs_data = GridSpecFromSubplotSpec(
        nrows=n_dof * 2, ncols=1, subplot_spec=gs_master[0, 1]
    )
    axes = [fig.add_subplot(gs_data[m, 0]) for m in range(n_dof * 2)]
    ax2d = fig.add_subplot(gs_2d[0, 0])
    ax2d.set_aspect("equal")
    for i, n in enumerate(random.sample(range(n_data), min(n_data, n_show))):
        # Compute reconstructed data
        trajectory = np.array(trajectories[n], dtype=np.float64)

        amplitudes, delays = synergy.encode(
            trajectory.tolist(), tvsynergies, n_activities_max
        )
        trajectory_est_list = synergy.decode(
            amplitudes, delays, tvsynergies, trajectory_length
        )
        trajectory_est = np.array(trajectory_est_list, dtype=np.float64)

        positions = np.cumsum(trajectory, axis=0)
        positions_est = np.cumsum(trajectory_est, axis=0)

        # Plot 2d position trajectory
        ax2d.plot(
            -(positions[:, 0] - positions[:, 0 + n_dof]),
            positions[:, 1] - positions[:, 1 + n_dof],
            lw=2,
            ls=":",
            color="C{}".format(i),
        )
        ax2d.plot(
            -(positions_est[:, 0] - positions_est[:, 0 + n_dof]),
            positions_est[:, 1] - positions_est[:, 1 + n_dof],
            lw=2,
            color="C{}".format(i),
        )

        # Plot time-series of velocity
        for m, ax in enumerate(axes):
            ax.plot(
                np.arange(len(trajectory)),
                trajectory[:, m],
                lw=2,
                ls=":",
                color="C{}".format(i),
            )
            ax.plot(
                np.arange(len(trajectory)),
                trajectory_est[:, m],
                lw=1,
                color="C{}".format(i),
            )
            ax.set_xlim((0, len(trajectory) - 1))

    # Plot extracted synergies
    gs_synergies = GridSpecFromSubplotSpec(
        nrows=n_synergies, ncols=1, subplot_spec=gs_master[0, 2]
    )
    for k in range(n_synergies):
        ax = fig.add_subplot(gs_synergies[k, 0])
        for m in range(n_dof):
            ax.plot(
                np.arange(synergy_length), synergies[k, :, m], color="C{}".format(m)
            )
            ax.plot(
                np.arange(synergy_length),
                synergies[k, :, m + n_dof],
                ls=":",
                color="C{}".format(m),
            )
        ax.set_xlim((0, synergy_length))

    filename_fig = os.path.join(synergy_dir, "synergies.pdf")
    fig.savefig(filename_fig)

    plt.show()


def convert_dataset(
    dataset: list[npt.NDArray[np.float64]],
) -> list[list[list[float]]]:
    """
    データセットを、時系列長が揃っていて正値化された速度軌道のリストに変換する

    Parameters
    ----------
    dataset: list[npt.NDArray[np.float64]]
        データセット

    Returns
    -------
    trajectories: list[list[list[float]]]
        速度軌道のリスト。時系列長が揃っていて、正値と負値とで次元が分けられている。
    """

    n_dim: Final[int] = 2
    n_markers: Final[int] = 27
    idx_start: Final[int] = 1 + 2 + n_markers
    n_data: Final[int] = len(dataset)

    # Compute the maximum trajectory length
    max_length = 0
    for data in dataset:
        length = data.shape[0]
        max_length = max(max_length, length)

    # Create equalized-length trajectories
    trajectories_array = np.zeros((n_data, max_length, n_dim * 2), dtype=np.float64)
    for i, data in enumerate(dataset):
        length = data.shape[0]
        data_positive = np.maximum(data[:, idx_start : idx_start + 2], 0.0)
        data_negative = np.maximum(-data[:, idx_start : idx_start + 2], 0.0)

        trajectories_array[i, 0:length, :n_dim] = data_positive
        trajectories_array[i, 0:length, n_dim:] = data_negative

    return trajectories_array.tolist()


def convert_activity(
    amplitudes: list[list[float]],
    delays: list[list[int]],
    trajectory_length: int,
) -> npt.NDArray[np.float64]:

    n_synergies: Final[int] = len(amplitudes)
    n_activities: Final[int] = len(amplitudes[0])

    activity = np.zeros((trajectory_length, n_synergies), dtype=np.float64)

    for i in range(n_synergies):
        for j in range(n_activities):
            amp = amplitudes[i][j]
            tau = delays[i][j]

            activity[tau, i] = amp

    return activity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-synergies", type=int, default=4)
    parser.add_argument("--synergy-length", type=int, default=25)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3.0)
    __args = Args(**vars(parser.parse_args()))

    evaluate(__args)
