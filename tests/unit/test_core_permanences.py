import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from pytorchimagepipeline.core.permanences import Device, ProgressManager, VRAMUsageError


@pytest.fixture
def mock_torch_cuda():
    with (
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.memory_reserved", side_effect=[100, 200]),
        patch(
            "torch.cuda.get_device_properties", side_effect=[MagicMock(total_memory=1000), MagicMock(total_memory=1000)]
        ),
    ):
        yield


def test_calculate_best_device(mock_torch_cuda):
    device = Device()
    assert device.device == torch.device("cuda:0")


def test_calculate_best_device_vram_usage_error(mock_torch_cuda):
    with patch("torch.cuda.memory_reserved", side_effect=[900, 950]), pytest.raises(VRAMUsageError):
        Device()


@pytest.fixture
def progress_manager():
    return ProgressManager(None, True)


def test_progress_manager(progress_manager, capsys):
    @progress_manager.progress_task("overall")
    def run(task_id, total, progress):
        for _ in range(total):
            progress.advance(task_id)
            time.sleep(0.1)

    with progress_manager.live:
        run(5)
    captured = capsys.readouterr()
    assert "overall" in captured.out
    assert "(5/5)" in captured.out
    assert "0:00:00" in captured.out
    assert captured.out.count("•") == 1


def test_progress_manager_with_status(progress_manager, capsys):
    progress_manager.progress_dict["overall"] = progress_manager._create_progress(with_status=True)
    progress_manager._init_live()

    @progress_manager.progress_task("overall")
    def run(task_id, total, progress):
        for idx in range(total):
            progress.advance(task_id)
            progress.update(task_id, status=f"{idx}/{total}")
            time.sleep(0.1)

    with progress_manager.live:
        run(5)
    captured = capsys.readouterr()
    assert "overall" in captured.out
    assert "(5/5)" in captured.out
    assert "0:00:00" in captured.out
    assert "• 4/5 •" in captured.out
    assert captured.out.count("•") == 2


def test_progress_manager_nested(progress_manager, capsys):
    @progress_manager.progress_task("overall")
    def run(task_id, total, progress):
        for _ in range(total):
            progress.advance(task_id)
            time.sleep(0.1)

    @progress_manager.progress_task("overall")
    def run_nested(task_id, total, progress):
        for _ in range(total):
            run(5)
            progress.advance(task_id)

    with progress_manager.live:
        run_nested(5)
    captured = capsys.readouterr()
    assert len(captured.out.split("\n")) == 6
    for each_line in captured.out.split("\n"):
        assert "overall" in each_line
        assert "(5/5)" in each_line
        assert "0:00:00" in each_line


def test_progress_manager_wrong_total(progress_manager, capsys):
    @progress_manager.progress_task("overall")
    def run(task_id, total, progress):
        for _ in range(total):
            progress.update(task_id, advance=0.5)
            time.sleep(0.1)

    with progress_manager.live:
        run(5)
    captured = capsys.readouterr()
    assert "overall" in captured.out
    assert "(2.5/5)" in captured.out
    assert "0:00:01" in captured.out


def test_progress_manager_not_provided(progress_manager):
    @progress_manager.progress_task("not_provided")
    def run(task_id, total, progress):
        for _ in range(total):
            progress.update(task_id, advance=0.5)
            time.sleep(0.1)

    with progress_manager.live, pytest.raises(NotImplementedError):
        run(5)
