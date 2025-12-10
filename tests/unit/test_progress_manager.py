from unittest.mock import MagicMock

import pytest
from rich.console import Console

from tipi.core.permanences import ProgressManager


@pytest.fixture
def mock_console():
    return MagicMock()


@pytest.fixture
def progress_manager():
    console = Console(force_terminal=True, soft_wrap=True, color_system="truecolor")
    return ProgressManager(console=console)


def test_init(progress_manager):
    assert isinstance(progress_manager.progress_dict, dict)
    assert "overall" in progress_manager.progress_dict
    assert "cleanup" in progress_manager.progress_dict
    assert "result" in progress_manager.progress_dict


def test_add_progress(progress_manager):
    progress_manager.add_progress("test_progress")
    assert "test_progress" in progress_manager.progress_dict


# TODO: Find a way to get capsys working
def test_add_task_to_progress(progress_manager, capsys):
    progress_manager.add_progress("test_progress")
    task_id = progress_manager.add_task_to_progress("test_progress", total=10)
    assert isinstance(task_id, int)
    assert not progress_manager.progress_dict["test_progress"].tasks[task_id].visible
    task_id = progress_manager.add_task_to_progress("test_progress", total=10, visible=True)
    assert isinstance(task_id, int)
    assert progress_manager.progress_dict["test_progress"].tasks[task_id].visible


def test_advance(progress_manager):
    progress_manager.add_progress("test_progress")
    task_id = progress_manager.add_task_to_progress("test_progress", total=10, visible=True)
    progress_manager.init_live()
    with progress_manager.live:
        progress_manager.advance("test_progress", task_id, step=2)
    progress = progress_manager.progress_dict["test_progress"]
    assert progress._tasks[task_id].completed == 2


def test_reset(progress_manager, capsys):
    # Clear Any Value from stdout
    capsys.readouterr()
    progress_manager.add_progress("test_progress")
    task_id = progress_manager.add_task_to_progress("test_progress", total=10, visible=True)
    progress_manager.init_live()
    with progress_manager.live:
        terminal_result = capsys.readouterr()
        assert "test_progress" in terminal_result.out
        progress_manager.advance("test_progress", task_id, step=5)
        terminal_result = capsys.readouterr()
        progress_manager.reset("test_progress")
    progress = progress_manager.progress_dict["test_progress"]
    assert progress._tasks[task_id].completed == 0


def test_get_progress_for_task(progress_manager):
    progress_manager.add_progress("test_progress")
    progress = progress_manager._get_progress_for_task("test_progress")
    assert progress is progress_manager.progress_dict["test_progress"]


def test_get_progress_for_task_invalid(progress_manager):
    with pytest.raises(RuntimeError):
        progress_manager._get_progress_for_task("invalid_task")
