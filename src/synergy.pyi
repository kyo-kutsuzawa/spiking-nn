class TimeVaryingSynergy:
    synergies: list[list[list[float]]]
    def __init__(self, n_synergies: int, synergy_length: int, n_dim: float) -> None: ...
    def extract(
        self, trajectories: list[list[list[float]]], n_iter: int, lr: float
    ) -> None: ...
    def encode(
        self,
        trajectory: list[list[float]],
        amplitudes: list[list[float]],
        delays: list[list[int]],
    ) -> None: ...
    def decode(
        self,
        amplitudes: list[list[float]],
        delays: list[list[int]],
        trajectory: list[list[float]],
    ) -> None: ...
