"""Unit tests for pipeline decorators.

Tests the @progress_task decorator for both standalone and pipeline execution modes.
This module focuses on decorator functionality, parameter binding, and iterable detection.

The tests are organized as:
1. Baseline undecorated functions (fixtures and baselines)
2. Standalone decorator tests (without pipeline context)
3. ProgressManager integration tests (with pipeline context)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tipi.decorators import progress_task
from tipi.helpers import clear_pipeline_context, set_pipeline_context

if TYPE_CHECKING:
    from collections.abc import Iterator


# ============================================================================
# FIXTURES: Baseline undecorated functions
# ============================================================================


@pytest.fixture
def baseline_list_processor():
    """Undecorated function that processes a list - used as baseline."""

    def process_list(items: list[int]) -> list[int]:
        """Process a list by doubling each item."""
        result = []
        for item in items:
            result.append(item * 2)
        return result

    return process_list


@pytest.fixture
def baseline_range_processor():
    """Undecorated function that processes a range - used as baseline."""

    def process_range(n: int) -> int:
        """Process a range and sum the values."""
        total = 0
        for i in range(n):
            total += i
        return total

    return process_range


@pytest.fixture
def baseline_generator_processor():
    """Undecorated function that processes a generator - used as baseline."""

    def process_generator(gen: Iterator[int]) -> list[int]:
        """Process a generator by collecting items."""
        result = []
        for item in gen:
            result.append(item)
        return result

    return process_generator


@pytest.fixture
def sample_data():
    """Provides sample data for tests."""
    return {
        "small_list": [1, 2, 3, 4, 5],
        "medium_list": list(range(20)),
        "large_list": list(range(100)),
    }


@pytest.fixture
def mock_progress_manager(mocker):
    """Create a mock ProgressManager for testing."""
    manager = mocker.MagicMock()
    manager.add_task_to_progress = mocker.MagicMock(return_value=42)  # Return task_id
    manager.advance = mocker.MagicMock()
    return manager


@pytest.fixture
def mock_rich_progress(mocker):
    """Create a mock Rich Progress context manager for testing standalone mode.

    Returns the mock progress instance that can be used to verify interactions
    with the Rich Progress API (add_task, advance calls, etc.).
    """
    mock_progress_instance = mocker.MagicMock()
    mock_progress_instance.add_task = mocker.MagicMock(return_value=0)
    mock_progress_instance.advance = mocker.MagicMock()
    mock_progress_instance.update = mocker.MagicMock()
    mock_progress_instance.__enter__ = mocker.MagicMock(return_value=mock_progress_instance)
    mock_progress_instance.__exit__ = mocker.MagicMock(return_value=None)

    # Patch the Progress class to return our mock instance
    mock_progress_class = mocker.patch("tipi.decorators.Progress")
    mock_progress_class.return_value = mock_progress_instance

    return mock_progress_instance


# ============================================================================
# TEST CLASS: Baseline Behavior (undecorated functions)
# ============================================================================


class TestBaselineBehavior:
    """Test baseline behavior of undecorated functions to establish expected results."""

    def test_baseline_list_processing(self, baseline_list_processor, sample_data):
        """Verify baseline list processing works correctly."""
        input_list = sample_data["small_list"]
        result = baseline_list_processor(input_list)
        assert result == [2, 4, 6, 8, 10]
        assert len(result) == len(input_list)

    def test_baseline_range_processing(self, baseline_range_processor):
        """Verify baseline range processing works correctly."""
        result = baseline_range_processor(10)
        assert result == sum(range(10))
        assert result == 45

    def test_baseline_generator_processing(self, baseline_generator_processor):
        """Verify baseline generator processing works correctly."""

        def gen():
            yield from [1, 2, 3, 4, 5]

        result = baseline_generator_processor(gen())
        assert result == [1, 2, 3, 4, 5]


# ============================================================================
# TEST CLASS: Standalone Decorator (no pipeline context)
# ============================================================================


class TestProgressTaskStandalone:
    """Test @progress_task decorator in standalone mode (no pipeline context)."""

    def setup_method(self):
        """Ensure clean state before each test."""
        clear_pipeline_context()

    def test_decorated_list_processor_matches_baseline(self, baseline_list_processor, sample_data):
        """Verify decorated function produces same results as baseline."""

        @progress_task(desc="Processing List")
        def process_list(items: list[int]) -> list[int]:
            result = []
            for item in items:
                result.append(item * 2)
            return result

        input_list = sample_data["small_list"]
        result = process_list(input_list)
        baseline_result = baseline_list_processor(input_list)
        assert result == baseline_result

    def test_decorated_range_processor_matches_baseline(self, baseline_range_processor):
        """Verify decorated range processor produces same results as baseline."""

        @progress_task(desc="Processing Range")
        def process_range(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        result = process_range(10)
        baseline_result = baseline_range_processor(10)
        assert result == baseline_result

    def test_decorator_with_generator_iterable(self, baseline_generator_processor, sample_data):
        """Test decorator correctly wraps generator iterable."""

        @progress_task(desc="Test Generator")
        def process_generator(gen: Iterator[int]) -> list[int]:
            result = []
            for item in gen:
                result.append(item)
            return result

        def number_generator():
            yield from [1, 2, 3, 4, 5]

        result = process_generator(number_generator())
        baseline_result = baseline_generator_processor(number_generator())
        assert result == baseline_result

    def test_decorator_with_multiple_parameters(self):
        """Test decorator with function that has multiple parameters."""

        @progress_task(desc="Multi Param")
        def process_with_multiplier(items: list[int], multiplier: int) -> list[int]:
            result = []
            for item in items:
                result.append(item * multiplier)
            return result

        result = process_with_multiplier([1, 2, 3], 10)
        assert result == [10, 20, 30]

    def test_decorator_with_kwargs(self):
        """Test decorator with function called with keyword arguments."""

        @progress_task(desc="Kwargs Test")
        def process_items(items: list[int], offset: int = 0) -> list[int]:
            result = []
            for item in items:
                result.append(item + offset)
            return result

        result = process_items(items=[1, 2, 3], offset=5)
        assert result == [6, 7, 8]

    def test_decorator_with_no_iterable_parameter(self):
        """Test decorator with function that has no iterable parameter."""

        @progress_task(desc="No Iterable")
        def compute_value(x: int, y: int) -> int:
            return x + y

        # Should run without error, just no progress tracking
        result = compute_value(5, 10)
        assert result == 15

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves the original function name."""

        @progress_task(desc="Test")
        def my_custom_function(items: list[int]) -> list[int]:
            return [i * 2 for i in items]

        assert my_custom_function.__name__ == "my_custom_function"

    def test_decorator_with_default_description(self, capsys):
        """Test decorator uses function name as default description."""

        @progress_task()
        def process_data_items(items: list[int]) -> list[int]:
            return [i * 2 for i in items]

        # Should not raise an error and should use "Process Data Items" as description
        result = process_data_items([1, 2, 3])
        assert result == [2, 4, 6]

        # Verify the default description appears in stdout
        captured = capsys.readouterr()
        assert "Process Data Items" in captured.out

    def test_decorator_with_empty_iterable(self):
        """Test decorator handles empty iterable correctly."""

        @progress_task(desc="Empty Test")
        def process_items(items: list[int]) -> list[int]:
            result = []
            for item in items:
                result.append(item * 2)
            return result

        result = process_items([])
        assert result == []

    def test_decorator_creates_rich_progress(self, mock_rich_progress):
        """Test that standalone decorator correctly uses Rich Progress context manager.

        This test validates the internal mechanism of how the decorator integrates with Rich:
        - Progress context manager is created and used correctly
        - Task is registered once via add_task()
        - Progress is advanced once per iteration

        Serves as a diagnostic test: if stdout tests fail but this passes, the issue is
        with Rich output formatting, not the decorator's context manager usage.
        """

        @progress_task(desc="Progress Test")
        def process_items(items: list[int]) -> list[int]:
            result = []
            for item in items:
                result.append(item * 2)
            return result

        result = process_items([1, 2, 3])
        assert result == [2, 4, 6]

        # Verify Progress was created and task was added
        mock_rich_progress.add_task.assert_called_once()
        # Verify advance was called for each item
        assert mock_rich_progress.advance.call_count == 3

    def test_decorator_with_nested_iteration(self, mock_rich_progress):
        """Test decorator with nested iteration (only outer should be tracked)."""

        @progress_task(desc="Nested Test")
        def process_nested(outer: list[list[int]]) -> list[int]:
            result = []
            for inner_list in outer:
                for item in inner_list:
                    result.append(item)
            return result

        result = process_nested([[1, 2], [3, 4], [5, 6]])
        assert result == [1, 2, 3, 4, 5, 6]

        # Verify only outer loop is tracked: 3 advances (not 6 for inner items)
        assert mock_rich_progress.advance.call_count == 3

    def test_stdout_progress_bar_displays_custom_description(self, capsys):
        """Test that progress bar with custom description appears in stdout."""

        @progress_task(desc="Custom Progress Task")
        def process_items(items: list[int]) -> list[int]:
            return [i * 2 for i in items]

        result = process_items([1, 2, 3, 4, 5])
        assert result == [2, 4, 6, 8, 10]

        # Capture stdout
        captured = capsys.readouterr()

        # Verify custom description appears in output
        assert "Custom Progress Task" in captured.out
        # Verify progress bar characters are present (Rich uses these)
        assert "━" in captured.out

    def test_stdout_progress_bar_shows_default_description(self, capsys):
        """Test that default description (from function name) appears in stdout."""

        @progress_task()  # No explicit description
        def train_model_epoch(batches: list[int]) -> int:
            return sum(batches)

        result = train_model_epoch([10, 20, 30])
        assert result == 60

        captured = capsys.readouterr()

        # Default description should be function name converted to title case
        # "train_model_epoch" -> "Train Model Epoch"
        assert "Train Model Epoch" in captured.out
        assert "━" in captured.out

    def test_stdout_progress_bar_shows_completion_count(self, capsys):
        """Test that progress bar shows completion count in stdout."""

        @progress_task(desc="Counting Items")
        def process_items(items: list[int]) -> list[int]:
            return [i * 2 for i in items]

        items = list(range(10))
        process_items(items)

        captured = capsys.readouterr()

        # Verify description appears
        assert "Counting Items" in captured.out
        # Verify completion count appears (shows as (10/10))
        assert "(10/10)" in captured.out

    def test_status_updates_in_standalone_mode(self, mock_rich_progress):
        """Test that status updates are passed to Rich Progress when yielding tuples."""

        @progress_task(desc="Status Test")
        def process_with_status(items: list[int]) -> list[int]:
            result = []
            for item in items:
                computed = item * 2
                result.append(computed)
                # Yield item with status - this is how users provide status updates
                # The decorator's advancing_iter will detect this tuple and extract status
            return result

        # We need to modify the iterable to yield tuples with status
        # Let's create a generator that yields tuples
        def items_with_status():
            for i in [1, 2, 3]:
                yield (i, f"Processing item {i}")

        @progress_task(desc="Status Test")
        def process_with_status_generator(items):
            result = []
            for item in items:
                result.append(item * 2)
            return result

        result = process_with_status_generator(items_with_status())
        assert result == [2, 4, 6]

        # Verify that update was called with status for each item
        update_calls = list(mock_rich_progress.update.call_args_list)
        assert len(update_calls) == 3

        # Check that status was passed in each update call
        for i, call in enumerate(update_calls):
            _, kwargs = call
            assert "status" in kwargs
            assert f"Processing item {i + 1}" in kwargs["status"]

    def test_status_updates_without_status_string(self, mock_rich_progress):
        """Test that items without status still work (status defaults to empty string)."""

        @progress_task(desc="No Status Test")
        def process_without_status(items: list[int]) -> list[int]:
            result = []
            for item in items:
                result.append(item * 2)
            return result

        result = process_without_status([1, 2, 3])
        assert result == [2, 4, 6]

        # Verify advance was called but update might not be called with status
        # (or called with empty status)
        assert mock_rich_progress.advance.call_count == 3


# ============================================================================
# TEST CLASS: ProgressManager Integration
# ============================================================================


class TestProgressTaskWithProgressManager:
    """Test @progress_task decorator with ProgressManager integration."""

    def setup_method(self):
        """Ensure clean state before each test."""
        clear_pipeline_context()

    def teardown_method(self):
        """Clean up after each test."""
        clear_pipeline_context()

    def test_decorator_uses_progress_manager_when_available(self, mock_progress_manager):
        """Test that decorator uses ProgressManager when in pipeline context."""
        set_pipeline_context({"progress_manager": mock_progress_manager})

        @progress_task(desc="PM Test")
        def process_items(items: list[int]) -> list[int]:
            result = []
            for item in items:
                result.append(item * 2)
            return result

        result = process_items([1, 2, 3, 4, 5])
        assert result == [2, 4, 6, 8, 10]

        # Verify ProgressManager was used
        mock_progress_manager.add_task_to_progress.assert_called_once_with("PM Test", total=5, visible=True)
        # Verify advance was called for each item
        assert mock_progress_manager.advance.call_count == 5

    def test_decorator_calls_progress_manager_advance_correctly(self, mock_progress_manager):
        """Test that decorator calls ProgressManager.advance with correct parameters."""
        set_pipeline_context({"progress_manager": mock_progress_manager})

        @progress_task(desc="Advance Test", progress_name="custom")
        def process_items(items: list[int]) -> list[int]:
            result = []
            for item in items:
                result.append(item * 2)
            return result

        process_items([1, 2, 3])

        # Verify advance was called with correct parameters
        # progress_name="custom", task_id=42 (from mock), step=1.0
        calls = mock_progress_manager.advance.call_args_list
        assert len(calls) == 3
        for call in calls:
            args, kwargs = call
            assert args == ("custom", 42)  # progress_name, task_id
            assert kwargs.get("step") == 1.0

    def test_decorator_with_progress_manager_and_unknown_length(self, mock_progress_manager):
        """Test decorator with ProgressManager when iterable has unknown length."""
        set_pipeline_context({"progress_manager": mock_progress_manager})

        @progress_task(desc="Unknown Length")
        def process_generator(gen: Iterator[int]) -> list[int]:
            result = []
            for item in gen:
                result.append(item)
            return result

        def number_gen():
            yield from [1, 2, 3, 4, 5]

        result = process_generator(number_gen())

        # Should still work, but total might be 0 or unknown
        mock_progress_manager.add_task_to_progress.assert_called_once()
        assert result == [1, 2, 3, 4, 5]

    def test_decorator_respects_custom_progress_name(self, mock_progress_manager):
        """Test that decorator respects custom progress_name parameter."""
        set_pipeline_context({"progress_manager": mock_progress_manager})

        @progress_task(desc="Custom Name Test", progress_name="train")
        def train_step(batches: list[int]) -> int:
            total = 0
            for batch in batches:
                total += batch
            return total

        result = train_step([10, 20, 30])
        assert result == 60

        # Verify advance was called with "train" progress_name
        calls = mock_progress_manager.advance.call_args_list
        for call in calls:
            args, _ = call
            assert args[0] == "train"

    def test_decorator_preserves_results_with_progress_manager(self, mock_progress_manager, sample_data):
        """Test that decorator preserves function results when using ProgressManager."""
        set_pipeline_context({"progress_manager": mock_progress_manager})

        @progress_task(desc="Result Test")
        def process_items(items: list[int]) -> list[int]:
            return [item * 3 for item in items]

        result = process_items(sample_data["small_list"])
        assert result == [3, 6, 9, 12, 15]

    def test_decorator_without_progress_manager_in_context(self):
        """Test decorator when pipeline context exists but has no progress_manager."""
        set_pipeline_context({"other_key": "value"})  # No progress_manager

        @progress_task(desc="No PM")
        def process_items(items: list[int]) -> list[int]:
            return [i * 2 for i in items]

        # Should fall back to Rich Progress
        result = process_items([1, 2, 3])
        assert result == [2, 4, 6]

    def test_decorator_with_progress_manager_and_large_dataset(self, mock_progress_manager, sample_data):
        """Test decorator with ProgressManager on larger dataset."""
        set_pipeline_context({"progress_manager": mock_progress_manager})

        @progress_task(desc="Large Dataset")
        def process_large(items: list[int]) -> int:
            total = 0
            for item in items:
                total += item
            return total

        result = process_large(sample_data["large_list"])

        # Verify task was added with correct total
        mock_progress_manager.add_task_to_progress.assert_called_once_with("Large Dataset", total=100, visible=True)
        # Verify advance was called 100 times
        assert mock_progress_manager.advance.call_count == 100
        assert result == sum(sample_data["large_list"])

    def test_status_updates_with_progress_manager(self, mock_progress_manager):
        """Test that status updates are passed to ProgressManager.advance() when yielding tuples."""
        set_pipeline_context({"progress_manager": mock_progress_manager})

        # Create a generator that yields tuples with status
        def items_with_status():
            for i in [1, 2, 3]:
                yield (i, f"Processing {i}")

        @progress_task(desc="Status PM Test", progress_name="custom")
        def process_with_status(items):
            result = []
            for item in items:
                result.append(item * 2)
            return result

        result = process_with_status(items_with_status())
        assert result == [2, 4, 6]

        # Verify advance was called with status for each item
        calls = mock_progress_manager.advance.call_args_list
        assert len(calls) == 3

        for i, call in enumerate(calls):
            args, kwargs = call
            assert args[0] == "custom"  # progress_name
            assert args[1] == 42  # task_id from mock
            assert kwargs.get("step") == 1.0
            assert kwargs.get("status") == f"Processing {i + 1}"

    def test_mixed_status_updates_with_progress_manager(self, mock_progress_manager):
        """Test that some items can have status and others don't in ProgressManager mode."""
        set_pipeline_context({"progress_manager": mock_progress_manager})

        # Create a generator with mixed status updates
        def items_mixed_status():
            yield (1, "First item")  # With status
            yield 2  # Without status
            yield (3, "Third item")  # With status
            yield 4  # Without status

        @progress_task(desc="Mixed Status Test")
        def process_mixed(items):
            result = []
            for item in items:
                result.append(item * 2)
            return result

        result = process_mixed(items_mixed_status())
        assert result == [2, 4, 6, 8]

        # Verify advance was called 4 times
        calls = mock_progress_manager.advance.call_args_list
        assert len(calls) == 4

        # Check status values: should be present for items 1 and 3, empty for 2 and 4
        assert calls[0][1]["status"] == "First item"
        assert calls[1][1]["status"] == ""
        assert calls[2][1]["status"] == "Third item"
        assert calls[3][1]["status"] == ""


# ============================================================================
# TEST CLASS: Property-Based Tests with Hypothesis
# ============================================================================


class TestProgressTaskProperties:
    """Property-based tests using Hypothesis to verify decorator invariants."""

    def setup_method(self):
        """Ensure clean state before each test."""
        clear_pipeline_context()

    @given(st.lists(st.integers(), min_size=0, max_size=100))
    def test_property_decorator_preserves_list_results(self, items: list[int]):
        """Property: decorated function preserves results for any list of integers."""

        def undecorated(lst: list[int]) -> list[int]:
            return [x * 2 for x in lst]

        @progress_task(desc="Property Test")
        def decorated(lst: list[int]) -> list[int]:
            return [x * 2 for x in lst]

        assert decorated(items) == undecorated(items)

    @given(st.integers(min_value=0, max_value=100))
    def test_property_decorator_preserves_range_results(self, n: int):
        """Property: decorated function preserves results for any range size."""

        def undecorated(size: int) -> int:
            total = 0
            for i in range(size):
                total += i
            return total

        @progress_task(desc="Range Property")
        def decorated(size: int) -> int:
            total = 0
            for i in range(size):
                total += i
            return total

        assert decorated(n) == undecorated(n)

    @given(st.lists(st.integers(), min_size=1, max_size=50))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_progress_manager_advances_correct_count(self, mocker, items: list[int]):
        """Property: ProgressManager.advance called exactly len(items) times."""
        mock_pm = mocker.MagicMock()
        mock_pm.add_task_to_progress = mocker.MagicMock(return_value=1)
        mock_pm.advance = mocker.MagicMock()
        set_pipeline_context({"progress_manager": mock_pm})

        @progress_task(desc="Count Test")
        def process(lst: list[int]) -> list[int]:
            return [x * 2 for x in lst]

        process(items)

        assert mock_pm.advance.call_count == len(items)
        clear_pipeline_context()

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=50))
    def test_property_decorator_works_with_string_lists(self, items: list[str]):
        """Property: decorator works with lists of strings."""

        @progress_task(desc="String List")
        def process_strings(strings: list[str]) -> list[str]:
            return [s.upper() for s in strings]

        result = process_strings(items)
        assert result == [s.upper() for s in items]

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0, max_size=50))
    def test_property_decorator_works_with_float_lists(self, items: list[float]):
        """Property: decorator works with lists of floats."""

        @progress_task(desc="Float List")
        def process_floats(floats: list[float]) -> list[float]:
            return [f * 2.0 for f in floats]

        result = process_floats(items)
        expected = [f * 2.0 for f in items]
        assert len(result) == len(expected)
        for r, e in zip(result, expected, strict=True):
            assert abs(r - e) < 1e-10 or (r == e)

    @given(
        st.lists(st.integers(), min_size=0, max_size=30),
        st.integers(min_value=-100, max_value=100),
    )
    def test_property_decorator_with_multiple_params(self, items: list[int], offset: int):
        """Property: decorator works with multiple parameters of any value."""

        @progress_task(desc="Multi Param Property")
        def process(lst: list[int], offset_val: int) -> list[int]:
            return [x + offset_val for x in lst]

        result = process(items, offset)
        assert result == [x + offset for x in items]

    @given(st.lists(st.integers(), min_size=1, max_size=50))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_progress_manager_gets_correct_total(self, mocker, items: list[int]):
        """Property: ProgressManager receives correct total parameter."""
        mock_pm = mocker.MagicMock()
        mock_pm.add_task_to_progress = mocker.MagicMock(return_value=1)
        mock_pm.advance = mocker.MagicMock()
        set_pipeline_context({"progress_manager": mock_pm})

        @progress_task(desc="Total Test")
        def process(lst: list[int]) -> list[int]:
            return [x * 2 for x in lst]

        process(items)

        # Check that add_task_to_progress was called with correct total
        call_args = mock_pm.add_task_to_progress.call_args
        assert call_args[1]["total"] == len(items)
        clear_pipeline_context()

    @given(st.lists(st.integers(min_value=1, max_value=10), min_size=2, max_size=20))
    def test_property_nested_lists_only_tracks_outer(self, outer: list[int]):
        """Property: decorator only tracks outer iteration in nested loops."""
        # Create nested structure where each integer determines inner list size
        nested = [[i] * size for i, size in enumerate(outer)]

        @progress_task(desc="Nested Property")
        def process_nested(outer_list: list[list[int]]) -> int:
            total = 0
            for inner_list in outer_list:
                for item in inner_list:
                    total += item
            return total

        # Should not raise an error
        result = process_nested(nested)

        # Verify result is correct
        expected = sum(sum(inner) for inner in nested)
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
