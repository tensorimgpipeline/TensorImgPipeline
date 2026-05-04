from unittest.mock import patch

import pyperf
from rich.console import Console
from rich.progress import Progress

from tipi.decorators import progress_task


# 1. Create a "Silent" version of Progress
class SilentProgress(Progress):
    def __init__(self, *args, **kwargs):
        # Force the console to be a dummy that does nothing
        with open("/dev/null", "w") as console_file:
            kwargs["console"] = Console(quiet=True, file=console_file)
            super().__init__(*args, **kwargs)


def baseline(data: list[int]) -> int:
    result = 0
    for i in data:
        result += i
    return result


@progress_task()
def decorated(data: list[int]) -> int:
    result = 0
    for i in data:
        result += i
    return result


def manual(data: list[int]) -> int:
    with Progress(console=Console(quiet=True)) as progress:
        tid = progress.add_task("Manual", total=len(data))

        result = 0
        for i in data:
            result += i
            progress.advance(tid, 1)
        return result


def run_bench():
    # pyperf.Runner() automatically handles -o/--output and --append via sys.argv
    runner = pyperf.Runner()

    # Define the sizes of the iterables to test
    sizes = [100, 1_000, 10_000, 100_000]

    with patch("tipi.decorators.Progress", SilentProgress):
        for size in sizes:
            data = list(range(size))

            # We use a unique name for each combination so pyperf can track them
            runner.bench_func(f"Baseline (size={size})", baseline, data)
            runner.bench_func(f"Decorated (size={size})", decorated, data)
            runner.bench_func(f"Manual (size={size})", manual, data)


if __name__ == "__main__":
    run_bench()
