import optuna

from train_synergy_generator import Args, train_synergy_model


def main():
    study = optuna.create_study(direction=optuna.study.StudyDirection.MINIMIZE)
    study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)

    print(study.best_params)


def objective(trial: optuna.Trial) -> float:
    args = Args(
        id=0,
        gain_in=0.0,
        gain_out=trial.suggest_float("gain_out", 0.1, 10.0, log=True),
        train_interval=trial.suggest_int("train_interval", 1, 20),
        # n_units=trial.suggest_int("n_units", 500, 3000),
        n_units=1000,
        connection_ratio_x=trial.suggest_float(
            "connection_ratio_x", 0.001, 0.9, log=True
        ),
        # connection_ratio_in=trial.suggest_float(
        #     "connection_ratio_in", 0.001, 0.9, log=True
        # ),
        connection_ratio_in=0.01,
        alpha=trial.suggest_float("alpha", 0.1, 100.0, log=True),
        G=trial.suggest_float("G", 1.0, 5e4, log=True),
        Q=trial.suggest_float("Q", 1.0, 5e4, log=True),
        bias=trial.suggest_float("bias", 1.0, 5000.0, log=True),
        t_end=25.0,
        t_train_start=0.0,
        t_train_finish=15.0,
        show_progress=False,
        plot=False,
    )

    loss = train_synergy_model(args)

    return loss


if __name__ == "__main__":
    main()
