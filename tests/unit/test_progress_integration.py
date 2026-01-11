"""Unit tests for progress_bar helper and progress_task decorator.

Tests both standalone mode (no pipeline context) and pipeline mode (with ProgressManager).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest

from tipi.decorators import progress_task
from tipi.helpers import clear_pipeline_context, progress_bar, set_pipeline_context

if TYPE_CHECKING:
    from collections.abc import Iterator


class TestProgressBarStandalone:
    """Test progress_bar helper in standalone mode (no pipeline context)."""

    def setup_method(self) -> None:
        """Ensure clean state before each test."""
        clear_pipeline_context()

    def test_progress_bar_with_list(self, capsys: pytest.CaptureFixture) -> None:
        """Test progress_bar with a simple list and verify console output."""
        items = [1, 2, 3, 4, 5]
        result = []

        for item in progress_bar(items, desc="Test"):
            result.append(item)

        assert result == items

        # Verify Rich progress output was written to stdout
        captured = capsys.readouterr()
        # Rich writes ANSI escape codes and progress updates
        assert "Test" in captured.out
        assert captured.out.count("━") == 40

    def test_progress_bar_with_range(self, capsys: pytest.CaptureFixture) -> None:
        """Test progress_bar with range object and verify output."""
        result = []

        for i in progress_bar(range(10), desc="Range Test"):
            result.append(i)

        assert result == list(range(10))

        # Verify progress description appears in output
        captured = capsys.readouterr()
        assert "Range Test" in captured.out
        assert captured.out.count("━") == 40

    def test_progress_bar_with_custom_total(self, capsys: pytest.CaptureFixture) -> None:
        """Test progress_bar with explicitly provided total and verify output."""
        items = [1, 2, 3]

        result = []
        for item in progress_bar(items, desc="Custom Total", total=10):
            result.append(item)

        assert result == items

        # Verify custom description appears
        captured = capsys.readouterr()
        assert "Custom Total" in captured.out
        assert captured.out.count("━") == 12

    @patch("tipi.helpers.track")
    def test_progress_bar_uses_rich_track_standalone(self, mock_track: Mock) -> None:
        """Verify progress_bar uses rich.track in standalone mode."""
        items = [1, 2, 3]
        mock_track.return_value = iter(items)

        list(progress_bar(items, desc="Test", total=3))

        mock_track.assert_called_once_with(items, description="Test", total=3)


class TestProgressBarWithPipeline:
    """Test progress_bar helper with pipeline ProgressManager."""

    def setup_method(self) -> None:
        """Setup mock ProgressManager for each test."""
        clear_pipeline_context()
        self.mock_progress_mgr = MagicMock()
        self.mock_progress_mgr.add_task_to_progress.return_value = 123  # task_id

    def teardown_method(self) -> None:
        """Clean up pipeline context after each test."""
        clear_pipeline_context()

    def test_progress_bar_with_progress_manager(self) -> None:
        """Test progress_bar uses ProgressManager when in pipeline context."""
        set_pipeline_context({"progress_manager": self.mock_progress_mgr})

        items = [1, 2, 3, 4, 5]
        result = []

        for item in progress_bar(items, desc="Pipeline Test"):
            result.append(item)

        # Verify task was added
        self.mock_progress_mgr.add_task_to_progress.assert_called_once_with("Pipeline Test", total=5, visible=True)

        # Verify advance was called for each item
        assert self.mock_progress_mgr.advance.call_count == 5
        for call in self.mock_progress_mgr.advance.call_args_list:
            args, kwargs = call
            assert args == ("overall", 123)
            assert kwargs == {"step": 1.0}

        assert result == items

    def test_progress_bar_custom_progress_name(self) -> None:
        """Test progress_bar with custom progress_name parameter."""
        set_pipeline_context({"progress_manager": self.mock_progress_mgr})

        items = [1, 2, 3]
        list(progress_bar(items, desc="Custom", progress_name="train"))

        # Verify advance uses custom progress_name
        for call in self.mock_progress_mgr.advance.call_args_list:
            args, _ = call
            assert args[0] == "train"


class TestProgressTaskDecoratorStandalone:
    """Test @progress_task decorator in standalone mode."""

    def setup_method(self) -> None:
        """Ensure clean state before each test."""
        clear_pipeline_context()

    def test_progress_task_with_list(self, capsys: pytest.CaptureFixture) -> None:
        """Test @progress_task decorator with list parameter and verify output."""

        @progress_task(desc="Processing Items")
        def process_items(items: list[int]) -> int:
            total = 0
            for item in items:
                total += item
            return total

        items = [1, 2, 3, 4, 5]
        result = process_items(items)

        assert result == 15

        # Verify progress output
        captured = capsys.readouterr()
        assert "Processing Items" in captured.out or "Processing Items" in captured.err

    def test_progress_task_with_range(self, capsys: pytest.CaptureFixture) -> None:
        """Test @progress_task decorator with range and verify output."""

        @progress_task(desc="Processing Range")
        def process_range(count: range) -> list[int]:
            result = []
            for i in count:
                result.append(i * 2)
            return result

        result = process_range(range(5))
        assert result == [0, 2, 4, 6, 8]

        # Verify progress output
        captured = capsys.readouterr()
        assert "Processing Range" in captured.out or "Processing Range" in captured.err

    def test_progress_task_auto_description(self, capsys: pytest.CaptureFixture) -> None:
        """Test @progress_task with auto-generated description from function name."""

        @progress_task()  # No desc provided
        def train_model(data: list[int]) -> int:
            return sum(data)

        # Should use "Train Model" as description
        result = train_model([1, 2, 3])
        assert result == 6

        # Verify auto-generated description appears
        captured = capsys.readouterr()
        assert "Train Model" in captured.out or "Train Model" in captured.err

    def test_progress_task_with_method(self, capsys: pytest.CaptureFixture) -> None:
        """Test @progress_task decorator on a class method and verify output."""

        class Trainer:
            def __init__(self) -> None:
                self.total = 0

            @progress_task(desc="Training Epoch")
            def train_epoch(self, dataloader: list[int]) -> int:
                for batch in dataloader:
                    self.total += batch
                return self.total

        trainer = Trainer()
        result = trainer.train_epoch([1, 2, 3, 4, 5])

        assert result == 15
        assert trainer.total == 15

        # Verify progress output
        captured = capsys.readouterr()
        assert "Training Epoch" in captured.out or "Training Epoch" in captured.err

    def test_progress_task_with_no_iterable(self) -> None:
        """Test @progress_task when function has no iterable parameter."""

        @progress_task(desc="No Iterable")
        def compute(x: int, y: int) -> int:
            return x + y

        # Should run normally without progress
        result = compute(5, 10)
        assert result == 15

    def test_progress_task_with_enumerate(self) -> None:
        """Test @progress_task with enumerate() as iterable."""

        @progress_task(desc="Enumerated Processing")
        def process_with_index(data_enum: enumerate) -> list[tuple[int, int]]:
            result = []
            for idx, value in data_enum:
                result.append((idx, value * 2))
            return result

        data = [10, 20, 30]
        result = process_with_index(enumerate(data))

        assert result == [(0, 20), (1, 40), (2, 60)]


class TestProgressTaskWithPipeline:
    """Test @progress_task decorator with pipeline ProgressManager."""

    def setup_method(self) -> None:
        """Setup mock ProgressManager for each test."""
        clear_pipeline_context()
        self.mock_progress_mgr = MagicMock()
        self.mock_progress_mgr.add_task_to_progress.return_value = 456  # task_id

    def teardown_method(self) -> None:
        """Clean up pipeline context after each test."""
        clear_pipeline_context()

    def test_progress_task_uses_progress_manager(self) -> None:
        """Test @progress_task uses ProgressManager when in pipeline context."""
        set_pipeline_context({"progress_manager": self.mock_progress_mgr})

        @progress_task(desc="Pipeline Task", progress_name="train")
        def process_data(items: list[int]) -> int:
            total = 0
            for item in items:
                total += item
            return total

        items = [1, 2, 3, 4, 5]
        result = process_data(items)

        # Verify task was added
        self.mock_progress_mgr.add_task_to_progress.assert_called_once_with("Pipeline Task", total=5, visible=True)

        # Verify advance was called for each item
        assert self.mock_progress_mgr.advance.call_count == 5
        for call in self.mock_progress_mgr.advance.call_args_list:
            args, kwargs = call
            assert args == ("train", 456)
            assert kwargs == {"step": 1.0}

        assert result == 15

    def test_progress_task_method_in_pipeline(self) -> None:
        """Test @progress_task on method with pipeline context."""
        set_pipeline_context({"progress_manager": self.mock_progress_mgr})

        class Trainer:
            @progress_task(desc="Training", progress_name="overall")
            def train(self, data: list[int]) -> int:
                return sum(data)

        trainer = Trainer()
        result = trainer.train([10, 20, 30])

        assert result == 60
        self.mock_progress_mgr.add_task_to_progress.assert_called_once()
        assert self.mock_progress_mgr.advance.call_count == 3

    def test_progress_task_switches_to_standalone(self) -> None:
        """Test decorator switches between pipeline and standalone modes."""
        # First run with pipeline
        set_pipeline_context({"progress_manager": self.mock_progress_mgr})

        @progress_task(desc="Switchable")
        def process(items: list[int]) -> int:
            return sum(items)

        result1 = process([1, 2, 3])
        assert result1 == 6
        assert self.mock_progress_mgr.add_task_to_progress.called

        # Clear context and run standalone
        clear_pipeline_context()
        result2 = process([4, 5, 6])
        assert result2 == 15  # Should still work

    def test_progress_task_without_len_support(self) -> None:
        """Test @progress_task with iterable that doesn't support len()."""

        def custom_generator() -> Iterator[int]:
            yield 1
            yield 2
            yield 3

        @progress_task(desc="Generator Test")
        def process_gen(gen: Iterator[int]) -> int:
            total = 0
            for item in gen:
                total += item
            return total

        result = process_gen(custom_generator())
        assert result == 6


class TestProgressTaskAndProgressBarIntegration:
    """Test interaction between progress_bar and @progress_task."""

    def setup_method(self) -> None:
        """Setup for integration tests."""
        clear_pipeline_context()

    def test_nested_progress_contexts(self) -> None:
        """Test that nested progress contexts raise LiveError in standalone mode.

        Rich doesn't allow multiple Live displays at once. In standalone mode,
        both @progress_task and progress_bar use their own Progress contexts,
        which causes conflicts. In pipeline mode, they share the ProgressManager.
        """
        from rich.errors import LiveError

        @progress_task(desc="Outer Task")
        def outer_process(epochs: range) -> int:
            total = 0
            for epoch in epochs:
                # Inner progress_bar will conflict with outer decorator's Progress
                for i in progress_bar(range(5), desc=f"Epoch {epoch}"):
                    total += i
            return total

        # In standalone mode, nested progress contexts raise LiveError
        with pytest.raises(LiveError, match="Only one live display may be active at once"):
            outer_process(range(3))

    def test_nested_progress_works_in_pipeline_mode(self) -> None:
        """Test that nested progress contexts work in pipeline mode.

        When using ProgressManager, both decorator and helper share the same
        manager, so nested contexts work without conflicts.
        """
        mock_progress_mgr = MagicMock()
        mock_progress_mgr.add_task_to_progress.side_effect = [100, 200, 201, 202]  # task_ids
        set_pipeline_context({"progress_manager": mock_progress_mgr})

        @progress_task(desc="Outer Task", progress_name="overall")
        def outer_process(epochs: range) -> int:
            total = 0
            for _epoch in epochs:
                # Inner progress_bar shares the same ProgressManager
                for i in progress_bar(range(5), desc="Inner", progress_name="batch"):
                    total += i
            return total

        result = outer_process(range(3))

        # Should work without errors
        assert result == 30  # (0+1+2+3+4) * 3 epochs

        # Both decorator and helper should have added tasks
        assert mock_progress_mgr.add_task_to_progress.call_count == 4  # 1 outer + 3 inner

        clear_pipeline_context()

    def test_both_use_same_progress_manager(self) -> None:
        """Test that both helper and decorator use the same ProgressManager."""
        mock_progress_mgr = MagicMock()
        mock_progress_mgr.add_task_to_progress.return_value = 789
        set_pipeline_context({"progress_manager": mock_progress_mgr})

        @progress_task(desc="Decorated", progress_name="train")
        def process_with_helper(outer: list[int]) -> int:
            total = 0
            for _item in outer:
                # Use progress_bar inside decorated function
                for i in progress_bar([1, 2, 3], desc="Inner", progress_name="batch"):
                    total += i
            return total

        result = process_with_helper([1, 2])

        # Both should use the same progress manager
        assert mock_progress_mgr.add_task_to_progress.call_count >= 2
        assert result == 12  # (1+2+3) * 2

        clear_pipeline_context()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
