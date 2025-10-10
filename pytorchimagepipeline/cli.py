# type: ignore
# ruff: noqa

from pathlib import Path

import typer
from rich import print as rprint

from pytorchimagepipeline.builder import PipelineBuilder, get_objects_for_pipeline

app = typer.Typer()


@app.command()
def build_pipeline(
    pipeline_name: str = typer.Argument(default="sem2segnet", help="The name of the pipeline to build."),
) -> None:
    """
    Build and execute an image processing pipeline.
    """

    objects, error = get_objects_for_pipeline(pipeline_name)
    if error:
        rprint(f"[bold red]Error:[/bold red] [green]No Pipeline '{pipeline_name}' defined yet![/green]")
        rprint(f"[bold red]Caused by:[/bold red] [yellow]{error}[/yellow]")
        raise typer.Exit()

    # Initialize the PipelineBuilder
    builder = PipelineBuilder()

    # Register each class in the builder
    for key in objects:
        error = builder.register_class(key, objects[key])
        if error:
            raise error

    # Load the configuration file
    error = builder.load_config(Path(f"{pipeline_name}/execute_pipeline.toml"))
    if error:
        raise error

    # Build the pipeline
    controller, error = builder.build()
    if error:
        raise error

    # Run the pipeline
    controller.run()


if __name__ == "__main__":
    app()
