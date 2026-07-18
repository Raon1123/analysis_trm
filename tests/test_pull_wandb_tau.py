import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lab"
    / "sync"
    / "sigma-k-tau-grid"
    / "pull_wandb_tau.py"
)
SPEC = importlib.util.spec_from_file_location("pull_wandb_tau", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pull_wandb_tau = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pull_wandb_tau)


class FakeRun:
    state = "finished"

    def __init__(self):
        self.scan_history_keys = None
        self.history_keys = None

    def scan_history(self, *, keys):
        self.scan_history_keys = keys
        return [
            {"_step": 10, "all.exact_accuracy": 0.25},
            {"_step": 20, "all.exact_accuracy": 0.75},
        ]

    def history(self, *, keys, samples, pandas):
        self.history_keys = keys
        assert samples == 1000
        assert pandas is False
        return [
            {"_step": 5, "train/exact_accuracy": 0.5},
            {"_step": 15, "train/exact_accuracy": 1.0},
        ]


class FakeApi:
    def __init__(self, run):
        self.run = run

    def runs(self, path, *, filters):
        assert path == "entity/project"
        assert filters == {"display_name": "cell-name"}
        return [self.run]


def test_pull_run_uses_dotted_cloud_eval_key_and_returns_curve():
    run = FakeRun()

    curve, state = pull_wandb_tau.pull_run(
        FakeApi(run), "entity", "project", "cell-name"
    )

    assert pull_wandb_tau.TEST_KEY == "all.exact_accuracy"
    assert pull_wandb_tau.TRAIN_KEY == "train/exact_accuracy"
    assert run.scan_history_keys == ["_step", "all.exact_accuracy"]
    assert run.history_keys == ["train/exact_accuracy"]
    assert curve == [(10, 0.25, 0.5), (20, 0.75, 1.0)]
    assert state == "finished"
