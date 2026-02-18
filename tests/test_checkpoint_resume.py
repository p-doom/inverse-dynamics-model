import pickle
from pathlib import Path

from idm.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint


class _FakeOpt:
    def __init__(self):
        self.v = {"a": 1}

    def state_dict(self):
        return dict(self.v)

    def load_state_dict(self, d):
        self.v = dict(d)


class _FakeModel:
    def __init__(self):
        self.loaded = None

    def save_pretrained(self, out_dir: str):
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter.bin").write_bytes(b"adapter")

    def state_dict(self):
        return {"w": [1, 2, 3]}

    def load_state_dict(self, d):
        self.loaded = d


def test_save_and_find_latest_checkpoint(tmp_path: Path):
    model = _FakeModel()
    opt = _FakeOpt()
    sch = _FakeOpt()
    save_checkpoint(
        out_dir=str(tmp_path),
        step_i=10,
        model=model,
        use_lora=True,
        optimizer=opt,
        scheduler=sch,
        scaler_state=None,
        train_state_d={"global_step": 10, "epoch_i": 2},
        grain_state_b=b"grain-state",
        args_d={"x": 1},
    )
    save_checkpoint(
        out_dir=str(tmp_path),
        step_i=20,
        model=model,
        use_lora=True,
        optimizer=opt,
        scheduler=sch,
        scaler_state={"scale": 2.0},
        train_state_d={"global_step": 20, "epoch_i": 3},
        grain_state_b=b"grain-state2",
        args_d={"x": 2},
    )
    latest = find_latest_checkpoint(str(tmp_path))
    assert latest is not None
    assert latest.endswith("step_00000020")


def test_load_checkpoint_restores_states(tmp_path: Path):
    model0 = _FakeModel()
    opt0 = _FakeOpt()
    sch0 = _FakeOpt()
    ckpt_dir = save_checkpoint(
        out_dir=str(tmp_path),
        step_i=7,
        model=model0,
        use_lora=False,
        optimizer=opt0,
        scheduler=sch0,
        scaler_state={"scale": 3.0},
        train_state_d={"global_step": 7, "epoch_i": 1},
        grain_state_b=b"state-7",
        args_d={"foo": "bar"},
    )
    model1 = _FakeModel()
    opt1 = _FakeOpt()
    sch1 = _FakeOpt()
    out_d = load_checkpoint(
        ckpt_dir=ckpt_dir,
        model=model1,
        use_lora=False,
        optimizer=opt1,
        scheduler=sch1,
    )
    assert out_d["train_state_d"]["global_step"] == 7
    assert out_d["grain_state_b"] == b"state-7"
    assert out_d["args_d"]["foo"] == "bar"
    assert isinstance(model1.loaded, dict)
    assert model1.loaded["w"] == [1, 2, 3]
    assert opt1.v == {"a": 1}
    assert sch1.v == {"a": 1}


def test_checkpoint_payload_is_pickle_readable(tmp_path: Path):
    ckpt_dir = save_checkpoint(
        out_dir=str(tmp_path),
        step_i=3,
        model=_FakeModel(),
        use_lora=True,
        optimizer=_FakeOpt(),
        scheduler=_FakeOpt(),
        scaler_state=None,
        train_state_d={"global_step": 3},
        grain_state_b=b"abc",
        args_d={"seed": 42},
    )
    payload_p = Path(ckpt_dir) / "trainer_state.pkl"
    payload_d = pickle.loads(payload_p.read_bytes())
    assert payload_d["train_state_d"]["global_step"] == 3
    assert payload_d["args_d"]["seed"] == 42
