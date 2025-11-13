from pathlib import Path
from typing import Optional

from pytorchimagepipeline.controller import PipelineController
from pytorchimagepipeline.core.executor import PipelineExecutor


class PipelineRunner:
    """High-level pipeline orchestrator.

    Coordinates the builder, controller, and executor.
    Handles WandB sweep integration.
    Provides programmatic API.
    """

    def __init__(self, pipeline_name: str, config_path: Path | None = None):
        self.pipeline_name = pipeline_name
        self.config_path = config_path or Path(f"{pipeline_name}/execute_pipeline.toml")

    def build(self) -> tuple[PipelineController, Optional[Exception]]:
        """Build the pipeline components."""
        # Get objects for pipeline
        # Create builder
        # Register classes
        # Load config
        # Build permanences & processes
        # Create controller
        # Return controller

    def run(self) -> None:
        """Execute the pipeline."""
        controller, error = self.build()
        if error:
            raise error

        # Check for WandB sweep
        wandb_logger = controller.get_permanence("wandb_logger", {})
        if wandb_logger and hasattr(wandb_logger, "sweep_id"):
            self._run_with_sweep(controller, wandb_logger)
        else:
            self._run_once(controller)

    def _run_once(self, controller: PipelineController) -> None:
        """Execute pipeline once."""
        executor = PipelineExecutor(controller)
        executor.run()

    def _run_with_sweep(self, controller: PipelineController, wandb_logger) -> None:
        """Execute pipeline with WandB sweep."""
        hyperparams = controller.get_permanence("hyperparams", {})
        wandb_logger.create_sweep(hyperparams.hyperparams.get("sweep_configuration", {}))
        wandb_logger.create_sweep_agent(lambda: self._run_once(controller))
