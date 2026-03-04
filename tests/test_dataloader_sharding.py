import pickle
from pathlib import Path

import numpy as np
import pytest
from array_record.python.array_record_module import ArrayRecordWriter

from idm.utils.data import count_source_records, get_dataloader


def _write_dummy_arrayrecord(
    path: Path, n_records: int = 12, T: int = 4, H: int = 2, W: int = 2, C: int = 3
):
    writer = ArrayRecordWriter(str(path), "group_size:1")
    for r_i in range(n_records):
        frames_THWC = np.full((T, H, W, C), r_i % 255, dtype=np.uint8)
        rec_d = {
            "raw_video": frames_THWC.tobytes(),
            "sequence_length": T,
            "path": f"r{r_i}.mp4",
            "actions": [f"a{r_i}_{t_i}" for t_i in range(T)],
        }
        writer.write(pickle.dumps(rec_d))
    writer.close()


def _keys_from_batch(batch_d):
    keys = batch_d["target_text"]
    if isinstance(keys, np.ndarray):
        return {str(x) for x in keys.tolist()}
    return {str(x) for x in keys}


def test_ddp_shards_are_disjoint(tmp_path: Path):
    p = tmp_path / "d.array_record"
    _write_dummy_arrayrecord(p)
    dl0 = get_dataloader(
        array_record_paths=[str(p)],
        seq_len=4,
        global_batch_size=4,
        image_h=2,
        image_w=2,
        image_c=3,
        rank=0,
        world_size=2,
        seed=123,
        epoch_i=0,
        num_workers=0,
        prefetch_buffer_size=1,
    )
    dl1 = get_dataloader(
        array_record_paths=[str(p)],
        seq_len=4,
        global_batch_size=4,
        image_h=2,
        image_w=2,
        image_c=3,
        rank=1,
        world_size=2,
        seed=123,
        epoch_i=0,
        num_workers=0,
        prefetch_buffer_size=1,
    )
    b0 = next(iter(dl0))
    b1 = next(iter(dl1))
    assert _keys_from_batch(b0).isdisjoint(_keys_from_batch(b1))


def test_epoch_shuffle_is_deterministic(tmp_path: Path):
    p = tmp_path / "d.array_record"
    _write_dummy_arrayrecord(p)
    kwargs = dict(
        array_record_paths=[str(p)],
        seq_len=4,
        global_batch_size=4,
        image_h=2,
        image_w=2,
        image_c=3,
        rank=0,
        world_size=2,
        seed=999,
        num_workers=0,
        prefetch_buffer_size=1,
    )
    e0_a = next(iter(get_dataloader(epoch_i=0, **kwargs)))
    e0_b = next(iter(get_dataloader(epoch_i=0, **kwargs)))
    e1 = next(iter(get_dataloader(epoch_i=1, **kwargs)))
    assert _keys_from_batch(e0_a) == _keys_from_batch(e0_b)
    assert _keys_from_batch(e0_a) != _keys_from_batch(e1)


def test_batch_size_must_divide_world_size(tmp_path: Path):
    p = tmp_path / "d.array_record"
    _write_dummy_arrayrecord(p)
    with pytest.raises(ValueError):
        get_dataloader(
            array_record_paths=[str(p)],
            seq_len=4,
            global_batch_size=3,
            image_h=2,
            image_w=2,
            image_c=3,
            rank=0,
            world_size=2,
            seed=0,
            epoch_i=0,
            num_workers=0,
            prefetch_buffer_size=1,
        )


def test_num_epochs_none_keeps_iterator_running(tmp_path: Path):
    p = tmp_path / "d.array_record"
    _write_dummy_arrayrecord(p, n_records=8)
    dl = get_dataloader(
        array_record_paths=[str(p)],
        seq_len=4,
        global_batch_size=4,
        image_h=2,
        image_w=2,
        image_c=3,
        rank=0,
        world_size=1,
        seed=7,
        epoch_i=0,
        num_epochs=None,
        num_workers=0,
        prefetch_buffer_size=1,
    )
    it = iter(dl)
    for _ in range(5):
        batch_d = next(it)
        assert len(batch_d["target_text"]) == 4


def test_min_action_density_must_be_in_unit_interval(tmp_path: Path):
    p = tmp_path / "d.array_record"
    _write_dummy_arrayrecord(p)
    with pytest.raises(ValueError):
        get_dataloader(
            array_record_paths=[str(p)],
            seq_len=4,
            global_batch_size=4,
            image_h=2,
            image_w=2,
            image_c=3,
            rank=0,
            world_size=1,
            seed=0,
            epoch_i=0,
            num_workers=0,
            prefetch_buffer_size=1,
            min_action_density=1.1,
        )


def test_action_upsample_random_fraction_must_be_in_unit_interval(tmp_path: Path):
    p = tmp_path / "d.array_record"
    _write_dummy_arrayrecord(p)
    with pytest.raises(ValueError):
        get_dataloader(
            array_record_paths=[str(p)],
            seq_len=4,
            global_batch_size=4,
            image_h=2,
            image_w=2,
            image_c=3,
            rank=0,
            world_size=1,
            seed=0,
            epoch_i=0,
            num_workers=0,
            prefetch_buffer_size=1,
            action_upsample_random_fraction=-0.1,
        )


def test_action_upsampling_r_zero_drops_pure_noop_sequences(tmp_path: Path):
    p = tmp_path / "d.array_record"
    writer = ArrayRecordWriter(str(p), "group_size:1")
    frames_THWC = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    writer.write(
        pickle.dumps(
            {
                "raw_video": frames_THWC.tobytes(),
                "sequence_length": 4,
                "path": "noop.mp4",
                "actions": ["NO_OP", "NO_OP", "NO_OP", "NO_OP"],
            }
        )
    )
    writer.write(
        pickle.dumps(
            {
                "raw_video": frames_THWC.tobytes(),
                "sequence_length": 4,
                "path": "active.mp4",
                "actions": [
                    "MOUSE:1,0,0",
                    "MOUSE:1,0,0",
                    "MOUSE:1,0,0",
                    "MOUSE:1,0,0",
                ],
            }
        )
    )
    writer.close()

    dl_no_filter = get_dataloader(
        array_record_paths=[str(p)],
        seq_len=4,
        global_batch_size=1,
        image_h=2,
        image_w=2,
        image_c=3,
        rank=0,
        world_size=1,
        seed=0,
        epoch_i=0,
        num_epochs=1,
        num_workers=0,
        prefetch_buffer_size=1,
        action_upsample_random_fraction=1.0,
    )
    all_texts = [str(batch_d["target_text"][0]) for batch_d in dl_no_filter]
    assert len(all_texts) == 2
    assert any("NO_OP" in text_s for text_s in all_texts)
    assert any("MOUSE:1,0,0" in text_s for text_s in all_texts)

    dl_upsampled = get_dataloader(
        array_record_paths=[str(p)],
        seq_len=4,
        global_batch_size=1,
        image_h=2,
        image_w=2,
        image_c=3,
        rank=0,
        world_size=1,
        seed=0,
        epoch_i=0,
        num_epochs=1,
        num_workers=0,
        prefetch_buffer_size=1,
        action_upsample_random_fraction=0.0,
    )
    kept_texts = [str(batch_d["target_text"][0]) for batch_d in dl_upsampled]
    assert len(kept_texts) == 1
    assert "MOUSE:1,0,0" in kept_texts[0]
    assert "NO_OP" not in kept_texts[0]


def test_count_source_records_matches_written_records(tmp_path: Path):
    p = tmp_path / "d.array_record"
    _write_dummy_arrayrecord(p, n_records=7)
    assert count_source_records([str(p)]) == 7
