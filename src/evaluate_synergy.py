import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from synergy import TimeVaryingSynergy


def example_decode() -> None:
    # Define constants
    n_synergies = 4
    synergy_length = 20
    n_dof = 2 * 2
    n_iter = 100
    lr = 0.01

    # Initialize time-varying synergies
    tvsynergies = TimeVaryingSynergy(n_synergies, synergy_length, n_dof)

    # Load a dataset
    dataset: list[npt.NDArray[np.float64]] = []
    datasets_dir = os.path.join(os.path.basename(__file__), "../dataset13/*.csv")
    filelist = glob.glob(datasets_dir)
    for filename in filelist:
        data = np.loadtxt(filename, delimiter=",")
        dataset.append(data)

    # Preprosessing the dataset
    trajectories: list[list[list[float]]] = []

    # Extract synergies
    tvsynergies.extract(trajectories, n_iter, lr)

    synergies = np.array(tvsynergies.synergies, dtype=np.float64)

    fig = plt.figure(figsize=(6, 4), constrained_layout=True)
    gs_master = GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[2, 1])

    gs_synergies = GridSpecFromSubplotSpec(
        nrows=n_synergies, ncols=1, subplot_spec=gs_master[0, 1]
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

    plt.show()


#     fig = plt.figure(figsize=(6, 4), constrained_layout=True)
#     gs_master = GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[2, 1])

#     # Plot reconstruction data
#     gs_1 = GridSpecFromSubplotSpec(nrows=M, ncols=1, subplot_spec=gs_master[0, 0])
#     axes = [fig.add_subplot(gs_1[m, 0]) for m in range(M)]
#     for n in range(N):
#         data = movements[n]
#         data_est = movements_est[n]
#         for m, ax in enumerate(axes):
#             ax.plot(np.arange(data.shape[0]), data[:, m], lw=2, ls=":", color="C{}".format(n))
#             ax.plot(np.arange(data.shape[0]), data_est[:, m], lw=1, color="C{}".format(n))
#             ax.set_xlim((0, data.shape[0] - 1))

#     # Plot synergy components
#     gs_2 = GridSpecFromSubplotSpec(nrows=K, ncols=1, subplot_spec=gs_master[0, 1])
#     axes2 = [fig.add_subplot(gs_2[k, 0]) for k in range(K)]
#     for k, ax in enumerate(axes2):
#         for m in range(M):
#             ax.plot(np.arange(args.synergy_length), synergies[k, :, m], color="C{}".format(m))
#             ax.plot(np.arange(args.synergy_length), synergies[k, :, m+M], ls=":", color="C{}".format(m))
#         ax.set_xlim((0, args.synergy_length))


# # Load a dataset
# dataset = []
# filelist = glob.glob(os.path.join(args.dirname, "*.csv"))
# filelist = sorted(filelist)
# for filename in filelist:
#     data = np.loadtxt(filename, delimiter=",")
#     dataset.append(data)

# # Extract movements
# n_markers = 8
# movements = [dataio.movement.read(data, "velocity", n_markers) for data in dataset]
# lengths = [mov.shape[0] for mov in movements]

# # Show data
# if args.show_original:
#     import matplotlib.pyplot as plt
#     fig, (ax1, ax2, ax3) = plt.subplots(2, 1, figsize=(4, 6), constrained_layout=True)
#     for i, d in enumerate(movements):
#         ax1.plot(np.arange(d.shape[0]), d[:, 0])
#         ax2.plot(np.arange(d.shape[0]), d[:, 1])
#     plt.show()

# movements = timevarying.transform_nonnegative(movements)
# dof = movements[0].shape[1]

# if args.synergies is None:
#     # Initialize synergies
#     synergies = np.random.uniform(0.0, 1.0, (args.n_synergies, args.synergy_length, dof))

#     # Extract motor synergies
#     refractory_period = int(args.synergy_length * 0.5)
#     for i in range(args.n_iter):
#         delays, amplitude = timevarying.match_synergies(movements, synergies, args.n_synergies_use, refractory_period)

#         r2 = timevarying.compute_R2(movements, synergies, amplitude, delays)
#         print("Iter {:4d}: R2 = {}".format(i, r2))

#         # Save synergies
#         np.save(os.path.join(args.out, "synergy.npy"), synergies)

#         synergies = timevarying.update_synergies(movements, synergies, amplitude, delays, args.lr)
# else:
#     synergies = np.load(args.synergies)
#     refractory_period = int(synergies.shape[1] * 0.5)
#     i = -1

# # Compute synergy activities
# delays, amplitude = timevarying.match_synergies(movements, synergies, args.n_synergies_use, refractory_period)
# r2 = timevarying.compute_R2(movements, synergies, amplitude, delays)
# print("Iter {:4d}: R2 = {}".format(i+1, r2))

# # Save results
# for n in range(len(dataset)):
#     data = dataset[n]

#     # Convert activities into time-series
#     activity = np.zeros((data.shape[0], args.n_synergies))
#     for k in range(args.n_synergies):
#         for ts, c in zip(delays[n][k], amplitude[n][k]):
#             activity[ts, k] = c

#     # Compute a residual term
#     movements_reconstruct = timevarying.decode([delays[n]], [amplitude[n]], synergies, [lengths[n]])[0]
#     movements_reconstruct = timevarying.inverse_transform_nonnegative([movements_reconstruct])[0]
#     pos = dataio.movement.read(dataset[n], "position", n_markers)
#     residual = np.zeros_like(pos)
#     for t in range(pos.shape[0] - 1):
#         residual[t] = (pos[t + 1] - pos[t]) - movements_reconstruct[t]

#     # Create a data
#     time = dataio.movement.read(data, "time", n_markers)
#     position = dataio.movement.read(data, "position", n_markers)
#     velocity = dataio.movement.read(data, "velocity", n_markers)
#     markers = dataio.movement.read(data, "markers", n_markers)
#     result = dataio.synergy.write(time, position, markers, velocity, activity, residual)

#     # Save to a csv file
#     filename = os.path.basename(filelist[n])
#     filename = os.path.join(args.out, filename)
#     np.savetxt(filename, result, delimiter=",")

# # Save synergies
# np.save(os.path.join(args.out, "synergy.npy"), synergies)

# # Save activities
# with open(os.path.join(args.out, "activity.pickle"), "wb") as f:
#     activities = (delays, amplitude)
#     pickle.dump(activities, f)

# if args.plot:
#     import matplotlib.pyplot as plt
#     from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

#     # Setup constants
#     N = len(movements)  # Number of data
#     M = movements[0].shape[1] // 2  # Number of DoF
#     K = args.n_synergies

#     # Reconstruct actions
#     lengths = [mov.shape[0] for mov in movements]
#     movements_est = timevarying.decode(delays, amplitude, synergies, lengths)
#     movements_est = timevarying.inverse_transform_nonnegative(movements_est)
#     movements = timevarying.inverse_transform_nonnegative(movements)

#     # Create a figure
#     fig = plt.figure(figsize=(6, 4), constrained_layout=True)
#     gs_master = GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[2, 1])

#     # Plot reconstruction data
#     gs_1 = GridSpecFromSubplotSpec(nrows=M, ncols=1, subplot_spec=gs_master[0, 0])
#     axes = [fig.add_subplot(gs_1[m, 0]) for m in range(M)]
#     for n in range(N):
#         data = movements[n]
#         data_est = movements_est[n]
#         for m, ax in enumerate(axes):
#             ax.plot(np.arange(data.shape[0]), data[:, m], lw=2, ls=":", color="C{}".format(n))
#             ax.plot(np.arange(data.shape[0]), data_est[:, m], lw=1, color="C{}".format(n))
#             ax.set_xlim((0, data.shape[0] - 1))

#     # Plot synergy components
#     gs_2 = GridSpecFromSubplotSpec(nrows=K, ncols=1, subplot_spec=gs_master[0, 1])
#     axes2 = [fig.add_subplot(gs_2[k, 0]) for k in range(K)]
#     for k, ax in enumerate(axes2):
#         for m in range(M):
#             ax.plot(np.arange(args.synergy_length), synergies[k, :, m], color="C{}".format(m))
#             ax.plot(np.arange(args.synergy_length), synergies[k, :, m+M], ls=":", color="C{}".format(m))
#         ax.set_xlim((0, args.synergy_length))

#     figname = os.path.join(args.out, "fig.pdf")
#     fig.savefig(figname)

#     plt.show()


if __name__ == "__main__":
    example_decode()
