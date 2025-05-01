from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pytorchimagepipeline.abstractions import (
    AbstractCombinedConfig,
    AbstractConfig,
    AbstractProgressManager,
    PipelineProcess,
    ProcessPlanType,
)
from pytorchimagepipeline.core.permanences import NullProgressManager, ProgressManager


def test_progress():
    class TestProcess(PipelineProcess):
        def __init__(self, manager, total):
            self.progress_manager = manager.progress or NullProgressManager()
            self.total = total

        def execute(self):
            self.progress_manager.add_task_to_progress("result", self.total)
            for _ in range(self.total):
                self.progress_manager.advance("result")

        def skip(self):
            return False

    @dataclass
    class TestTotalConfig(AbstractConfig):
        total: int

        def validate(self) -> None:
            if isinstance(self.total, int):
                ...

    @dataclass
    class TestConfig(AbstractCombinedConfig):
        config_file: Path | str

        config: dict[str, Any] = field(init=False)

        def __post_init__(self) -> None:
            self.test_config = TestTotalConfig(total=20)

    @dataclass
    class TestManager(AbstractProgressManager):
        def __parse_config__(self, config_file: Path) -> None:
            self.config = TestConfig(config_file=config_file)

        def __init_permanences__(self) -> None:
            # Core Init Permanences
            self.progress = ProgressManager()

        def __init_processes__(self) -> None:
            self.process_plan: ProcessPlanType = {
                "predict_masks": (TestProcess, self.config.test_config),
            }

    manager = TestManager(config_file=Path(""))
    manager.run()
