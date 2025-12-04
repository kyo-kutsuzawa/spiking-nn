class TimeVaryingSynergy:
    def __init__(
        self, n_synergies: int, synergy_length: int, n_dim: int, refractory_period: int
    ) -> None: ...
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
    def get_synergies(self) -> list[list[list[float]]]: ...

def extract(
    trajectories: list[list[list[float]]],
    n_synergies: int,
    synergy_length: int,
    refractory_period: int,
    n_activities_max: int,
    n_iter: int,
    lr: float,
) -> TimeVaryingSynergy: ...
def encode(
    trajectory: list[list[float]], synergies: TimeVaryingSynergy, n_activities_max: int
) -> tuple[list[list[float]], list[list[float]]]: ...
