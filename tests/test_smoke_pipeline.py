from pathlib import Path

import pytest

from idm.data import find_array_record_paths, get_dataloader


SMOKE_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-02-16/array_records_codex/smoke_codex_actions"
)


@pytest.mark.skipif(not SMOKE_ROOT.exists(), reason="smoke dataset path not available")
def test_smoke_dataset_yields_batch_with_in_record_actions():
    paths = find_array_record_paths(str(SMOKE_ROOT), "train")[:2]
    dl = get_dataloader(
        array_record_paths=paths,
        seq_len=32,
        global_batch_size=2,
        image_h=90,
        image_w=160,
        image_c=3,
        rank=0,
        world_size=1,
        seed=0,
        epoch_i=0,
        num_workers=0,
        prefetch_buffer_size=1,
    )
    batch_d = next(iter(dl))
    assert batch_d["frames"].shape[0] == 2
    assert batch_d["frames"].shape[1] == 32
    assert len(batch_d["target_text"]) == 2
