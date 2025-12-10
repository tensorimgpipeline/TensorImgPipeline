from __future__ import annotations

from pathlib import Path

from tipi.core.builder import PipelineBuilder, get_objects_for_pipeline
from tipi.core.controller import PipelineController
from tipi.core.executor import PipelineExecutor


class PipelineRunner:
    """High-level pipeline orchestrator.

    Coordinates the builder, controller, and executor.
    Handles WandB sweep integration.
    Provides programmatic API.
    """

    def __init__(self, pipeline_name: str, config_path: Path | None = None) -> None:
        self.pipeline_name = pipeline_name
        self.config_path = config_path or Path(f"{pipeline_name}/pipeline_config.toml")

    def build(self) -> PipelineController:
        """Build the pipeline components.

        Returns:
            PipelineController: Configured controller ready to execute

        Raises:
            ModuleNotFoundError: If pipeline module not found
            RegistryError: If class registration fails
            ConfigNotFoundError: If config file doesn't exist
            ConfigInvalidTomlError: If TOML parsing fails
            InstTypeError: If permanence/process instantiation fails
        """
        # Create a PipelineBuilder instance
        builder = PipelineBuilder()

        # First, try to load core permanences and processes (if they exist)
        try:
            core_objects = get_objects_for_pipeline("core")
            for class_name, class_type in core_objects.items():
                builder.register_class(class_name, class_type)
        except (ModuleNotFoundError, AttributeError):
            # Core module doesn't exist or doesn't have registries yet - that's okay
            # TODO this should create a warning or should be handled via cli.
            pass

        # Then load the pipeline-specific classes
        pipeline_objects = get_objects_for_pipeline(self.pipeline_name)
        for class_name, class_type in pipeline_objects.items():
            builder.register_class(class_name, class_type)

        # Load the configuration file
        builder.load_config(self.config_path)

        # Build permanences and processes
        permanences, process_specs = builder.build()

        # Create the controller with permanences and process specs
        controller = PipelineController(permanences, process_specs)

        return controller

    def run(self) -> None:
        """Execute the pipeline.

        Raises:
            Various exceptions from build() or execution
        """
        controller = self.build()

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

    def _run_with_sweep(self, controller: PipelineController, wandb_logger: object) -> None:
        """Execute pipeline with WandB sweep."""
        hyperparams = controller.get_permanence("hyperparams", {})
        wandb_logger.create_sweep(hyperparams.hyperparams.get("sweep_configuration", {}))  # type: ignore[attr-defined]
        wandb_logger.create_sweep_agent(lambda: self._run_once(controller))  # type: ignore[attr-defined]
