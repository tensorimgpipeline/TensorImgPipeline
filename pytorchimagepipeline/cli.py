"""CLI tool for PytorchImagePipeline.

Provides commands for running, inspecting, and managing ML pipelines.

Copyright (C) 2025 Matti Kaupenjohann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from pytorchimagepipeline.abstractions import Permanence, PipelineProcess
from pytorchimagepipeline.core.runner import PipelineRunner
from pytorchimagepipeline.paths import get_path_manager


def _exit_with_error(message: str, code: int = 1) -> None:
    """Print error message and exit.

    Args:
        message: Error message to display
        code: Exit code (default: 1)
    """
    rprint(message)
    raise typer.Exit(code=code)


app = typer.Typer(
    name="pytorchpipeline",
    help="PytorchImagePipeline CLI - From scripts to production pipelines",
    add_completion=False,
)
console = Console()

# Get path manager
path_manager = get_path_manager()


@app.command(name="run")
def run_pipeline(
    pipeline_name: str = typer.Argument(help="The name of the pipeline to run (e.g., 'sam2segnet')"),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to custom config file (relative to configs/)"
    ),
) -> None:
    """Run a pipeline by name.

    Examples:
        pytorchpipeline run sam2segnet
        pytorchpipeline run my_pipeline --config custom.toml
    """
    try:
        config_path = Path(config) if config else None
        runner = PipelineRunner(pipeline_name, config_path)
        runner.run()
    except Exception as err:
        rprint(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1) from err


@app.command(name="list")
def list_pipelines(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    show_links: bool = typer.Option(True, "--links/--no-links", help="Show symlink information"),
) -> None:
    """List all available pipelines.

    Shows both built-in pipelines and linked subpackages.

    Examples:
        pytorchpipeline list
        pytorchpipeline list -v
        pytorchpipeline list --no-links
    """
    pipelines_dir = path_manager.get_projects_dir()

    if not pipelines_dir.exists():
        rprint(f"[yellow]No pipelines directory found at {pipelines_dir}[/yellow]")
        rprint(f"\nMode: {path_manager.get_info()['mode']}")
        rprint("Create a pipeline with: [cyan]pytorchpipeline create my_pipeline[/cyan]")
        raise typer.Exit(code=1)

    # Find all pipeline directories
    pipeline_dirs = [d for d in pipelines_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]

    if not pipeline_dirs:
        rprint("[yellow]No pipelines found.[/yellow]")
        raise typer.Exit()

    # Create table
    table = Table(title="Available Pipelines")
    table.add_column("Pipeline", style="cyan", no_wrap=True)
    table.add_column("Type", style="yellow")
    table.add_column("Config", style="green")
    table.add_column("Status", style="white")

    if verbose:
        table.add_column("Permanences", style="blue")
        table.add_column("Processes", style="magenta")

    if show_links:
        table.add_column("Source", style="dim")

    for pipeline_dir in sorted(pipeline_dirs):
        pipeline_name = pipeline_dir.name
        config_path = path_manager.get_config_path(pipeline_name)

        # Check if it's a symlink
        is_symlink = pipeline_dir.is_symlink()
        pipeline_type = "Linked" if is_symlink else "Built-in"

        # Check if config exists
        has_config = "✓" if config_path.exists() else "✗"
        status = "Ready" if config_path.exists() else "Missing config"

        row_data = [pipeline_name, pipeline_type, has_config, status]

        if verbose:
            # Try to load pipeline module
            try:
                module = path_manager.import_project_module(pipeline_name)
                if module:
                    permanences = getattr(module, "permanences_to_register", {})
                    processes = getattr(module, "processes_to_register", {})

                    perm_count = len(permanences)
                    proc_count = len(processes)

                    row_data.extend([str(perm_count), str(proc_count)])
                else:
                    row_data.extend(["Error", "Error"])
            except Exception:
                row_data.extend(["Error", "Error"])

        if show_links:
            if is_symlink:
                target = pipeline_dir.resolve()
                row_data.append(str(target))
            else:
                row_data.append("-")

        table.add_row(*row_data)

    console.print(table)


@app.command(name="inspect")
def inspect_pipeline(
    pipeline_name: str = typer.Argument(help="Name of the pipeline to inspect"),
    show_docs: bool = typer.Option(False, "--docs", "-d", help="Show docstrings"),
) -> None:
    """Inspect a pipeline's components.

    Shows all permanences and processes registered for a pipeline.

    Examples:
        pytorchpipeline inspect sam2segnet
        pytorchpipeline inspect sam2segnet --docs
    """
    try:
        # Use path manager to import module
        module = path_manager.import_project_module(pipeline_name)
    except ImportError as err:
        rprint(f"[bold red]Error:[/bold red] Pipeline '{pipeline_name}' not found.")
        raise typer.Exit(code=1) from err

    try:
        permanences = getattr(module, "permanences_to_register", {})
        processes = getattr(module, "processes_to_register", {})

        # Create tree view
        tree = Tree(f"[bold cyan]{pipeline_name}[/bold cyan] Pipeline")

        # Add permanences
        if permanences:
            perm_branch = tree.add("[bold green]Permanences[/bold green]")
            for name, cls in permanences.items():
                cls_info = f"[cyan]{name}[/cyan]: {cls.__name__}"
                if show_docs and cls.__doc__:
                    doc = cls.__doc__.strip().split("\n")[0]
                    cls_info += f"\n  [dim]{doc}[/dim]"
                perm_branch.add(cls_info)
        else:
            tree.add("[yellow]No permanences registered[/yellow]")

        # Add processes
        if processes:
            proc_branch = tree.add("[bold magenta]Processes[/bold magenta]")
            for name, cls in processes.items():
                cls_info = f"[magenta]{name}[/magenta]: {cls.__name__}"
                if show_docs and cls.__doc__:
                    doc = cls.__doc__.strip().split("\n")[0]
                    cls_info += f"\n  [dim]{doc}[/dim]"
                proc_branch.add(cls_info)
        else:
            tree.add("[yellow]No processes registered[/yellow]")

        console.print(tree)

        # Show config location
        config_path = path_manager.get_config_path(pipeline_name, "execute_pipeline.toml")
        if config_path and config_path.exists():
            console.print(f"\n[green]✓[/green] Config: {config_path}")
        else:
            console.print(f"\n[red]✗[/red] Config not found: {config_path if config_path else 'N/A'}")

    except ModuleNotFoundError as err:
        rprint(f"[bold red]Error:[/bold red] Pipeline '{pipeline_name}' not found.")
        raise typer.Exit(code=1) from err
    except Exception as err:
        rprint(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1) from err


@app.command(name="create")
def create_project(
    project_name: str = typer.Argument(help="Name of the new pipeline project"),
    location: Optional[str] = typer.Option(
        None, "--location", "-l", help="Location to create project (default: current directory)"
    ),
    with_example: bool = typer.Option(False, "--example", "-e", help="Include example process and permanence"),
) -> None:
    """Create a new pipeline project with scaffolding.

    Creates a complete project structure for a new pipeline that can later
    be added to the main pipeline system.

    Examples:
        pytorchpipeline create my_pipeline
        pytorchpipeline create my_pipeline --location ./projects
        pytorchpipeline create my_pipeline --example
    """
    # Determine project location
    base_dir = Path(location) / project_name if location else Path.cwd() / project_name

    if base_dir.exists():
        rprint(f"[bold red]Error:[/bold red] Directory '{base_dir}' already exists.")
        raise typer.Exit(code=1)

    try:
        # Create project structure
        base_dir.mkdir(parents=True)
        src_dir = base_dir / project_name
        src_dir.mkdir()
        config_dir = base_dir / "configs"
        config_dir.mkdir()

        # Create package files
        (src_dir / "__init__.py").write_text(_generate_project_init(project_name, with_example))
        (src_dir / "permanences.py").write_text(_generate_permanence_file(project_name, with_example))
        (src_dir / "processes.py").write_text(_generate_processes_file(project_name, with_example))

        # Create config file
        (config_dir / "pipeline_config.toml").write_text(_generate_config_file(project_name, with_example))

        # Create README
        (base_dir / "README.md").write_text(_generate_readme(project_name))

        # Create pyproject.toml
        (base_dir / "pyproject.toml").write_text(_generate_pyproject(project_name))

        # Create .gitignore
        (base_dir / ".gitignore").write_text(_generate_gitignore())

        # Success message
        panel = Panel(
            f"[green]✓[/green] Pipeline project '{project_name}' created successfully!\n\n"
            f"Created at: {base_dir}\n\n"
            f"Structure:\n"
            f"  {project_name}/\n"
            f"  ├── {project_name}/\n"
            f"  │   ├── __init__.py\n"
            f"  │   ├── permanences.py\n"
            f"  │   └── processes.py\n"
            f"  ├── configs/\n"
            f"  │   └── pipeline_config.toml\n"
            f"  ├── pyproject.toml\n"
            f"  ├── README.md\n"
            f"  └── .gitignore\n\n"
            f"Next steps:\n"
            f"  1. cd {base_dir}\n"
            f"  2. Edit {project_name}/permanences.py and processes.py\n"
            f"  3. Update configs/pipeline_config.toml\n"
            f"  4. Link to main pipeline: pytorchpipeline add {base_dir}",
            title="Project Created",
            border_style="green",
        )
        console.print(panel)

    except Exception as err:
        rprint(f"[bold red]Error:[/bold red] Failed to create project: {err}")
        raise typer.Exit(code=1) from err


@app.command(name="add")
def add_subpackage(
    source: str = typer.Argument(help="Local path to project or Git repository URL"),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Name for the linked pipeline (defaults to directory name)"
    ),
    location: Optional[str] = typer.Option(
        None, "--location", "-l", help="Location to clone git repos (default: ./submodules)"
    ),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Git branch to checkout (if cloning)"),
) -> None:
    """Link an existing project or clone from git as a subpackage.

    This command either:
    1. Links a local project directory to the pipeline system
    2. Clones a git repository and links it

    Examples:
        # Link local project
        pytorchpipeline add ./my_pipeline_project
        pytorchpipeline add /path/to/project --name custom_name

        # Clone and link from git
        pytorchpipeline add https://github.com/user/ml-pipeline.git
        pytorchpipeline add git@github.com:user/pipeline.git --location ./external
        pytorchpipeline add https://github.com/user/repo.git --branch dev
    """
    import subprocess

    try:
        # Check if source is a git URL
        is_git_url = source.startswith(("https://", "http://", "git@", "git://"))

        if is_git_url:
            # Clone from git
            clone_location = Path(location) if location else Path("submodules")
            clone_location.mkdir(parents=True, exist_ok=True)

            # Extract repo name from URL if name not provided
            if not name:
                name = Path(source.rstrip("/")).stem.replace(".git", "")

            target_dir = clone_location / name

            if target_dir.exists():
                _exit_with_error(f"[bold red]Error:[/bold red] Target directory '{target_dir}' already exists.")

            console.print("[cyan]Cloning repository...[/cyan]")

            # Build git clone command
            cmd = ["git", "clone"]
            if branch:
                cmd.extend(["--branch", branch])
            cmd.extend([source, str(target_dir)])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                _exit_with_error(f"[bold red]Error:[/bold red] Git clone failed:\n{result.stderr}")

            console.print(f"[green]✓[/green] Repository cloned to {target_dir}")
            project_path = target_dir

        else:
            # Link local project
            project_path = Path(source).resolve()

            if not project_path.exists():
                _exit_with_error(f"[bold red]Error:[/bold red] Path '{source}' does not exist.")

            if not project_path.is_dir():
                _exit_with_error(f"[bold red]Error:[/bold red] Path '{source}' is not a directory.")

            if not name:
                name = project_path.name

        # Validate project structure
        package_dir = project_path / name
        if not package_dir.exists():
            # Try to find the package directory
            possible_dirs = [d for d in project_path.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
            if len(possible_dirs) == 1:
                package_dir = possible_dirs[0]
                name = package_dir.name
            else:
                rprint(f"[yellow]Warning:[/yellow] Could not find package directory in {project_path}")
                rprint("Expected structure: project_name/project_name/__init__.py")

        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            _exit_with_error("[bold red]Error:[/bold red] Not a valid Python package (missing __init__.py)")

        # Create symlink in projects directory
        projects_dir = path_manager.get_projects_dir()
        projects_dir.mkdir(parents=True, exist_ok=True)

        link_path = projects_dir / name

        if link_path.exists():
            if link_path.is_symlink():
                rprint(f"[yellow]Warning:[/yellow] Symlink already exists at {link_path}")
                overwrite = typer.confirm("Overwrite existing link?")
                if overwrite:
                    link_path.unlink()
                else:
                    _exit_with_error("", code=0)
            else:
                _exit_with_error(f"[bold red]Error:[/bold red] A directory/file already exists at {link_path}")

        # Create symlink
        link_path.symlink_to(package_dir.resolve(), target_is_directory=True)

        # Link config directory if it exists
        config_linked = False
        config_link_path = None
        project_config_dir = project_path / "configs"
        if project_config_dir.exists() and project_config_dir.is_dir():
            configs_dir = path_manager.get_configs_dir()
            configs_dir.mkdir(parents=True, exist_ok=True)

            config_link_path = configs_dir / name

            # Check if config link already exists
            if config_link_path.exists():
                if config_link_path.is_symlink():
                    config_link_path.unlink()
                else:
                    rprint(
                        f"[yellow]Warning:[/yellow] Config directory exists at {config_link_path}, not linking configs"
                    )

            if not config_link_path.exists():
                config_link_path.symlink_to(project_config_dir.resolve(), target_is_directory=True)
                config_linked = True

        # Success message
        config_msg = f"\n[green]✓[/green] Config linked to: {config_link_path}" if config_linked else ""
        panel = Panel(
            f"[green]✓[/green] Package '{name}' linked successfully!\n\n"
            f"Source: {project_path}\n"
            f"Linked to: {link_path}"
            f"{config_msg}\n\n"
            f"Next steps:\n"
            f"  1. Inspect: pytorchpipeline inspect {name}\n"
            f"  2. Validate: pytorchpipeline validate {name}\n"
            f"  3. Run: pytorchpipeline run {name}",
            title="Package Added",
            border_style="green",
        )
        console.print(panel)

    except subprocess.CalledProcessError as err:
        rprint(f"[bold red]Error:[/bold red] Command failed: {err}")
        raise typer.Exit(code=1) from err
    except Exception as err:
        rprint(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1) from err


@app.command(name="remove")
def remove_subpackage(
    name: str = typer.Argument(help="Name of the linked pipeline to remove"),
    delete_source: bool = typer.Option(
        False, "--delete-source", help="Also delete the source directory (if it was cloned)"
    ),
) -> None:
    """Remove a linked subpackage.

    Removes the symlink from the projects directory. Optionally deletes
    the source if it was cloned (in cache/).

    Examples:
        pytorchpipeline remove my_pipeline
        pytorchpipeline remove my_pipeline --delete-source
    """
    projects_dir = path_manager.get_projects_dir()
    link_path = projects_dir / name

    if not link_path.exists():
        rprint(f"[bold red]Error:[/bold red] Pipeline '{name}' not found.")
        raise typer.Exit(code=1)

    if not link_path.is_symlink():
        rprint(f"[bold red]Error:[/bold red] '{name}' is not a linked package (it's built-in).")
        rprint("Built-in pipelines cannot be removed this way.")
        raise typer.Exit(code=1)

    try:
        # Get the source path before unlinking
        source_path = link_path.resolve()

        # Remove project symlink
        link_path.unlink()
        console.print(f"[green]✓[/green] Removed link: {link_path}")

        # Remove config symlink if it exists
        configs_dir = path_manager.get_configs_dir()
        config_link_path = configs_dir / name
        if config_link_path.exists() and config_link_path.is_symlink():
            config_link_path.unlink()
            console.print(f"[green]✓[/green] Removed config link: {config_link_path}")

        # Optionally delete source
        if delete_source:
            # Check if source is in cache/ (safe to delete)
            cache_dir = path_manager.get_cache_dir().resolve()
            if source_path.is_relative_to(cache_dir):
                import shutil

                confirm = typer.confirm(f"Are you sure you want to delete {source_path}?", default=False)
                if confirm:
                    shutil.rmtree(source_path)
                    console.print(f"[green]✓[/green] Deleted source: {source_path}")
                else:
                    console.print("[yellow]Source deletion cancelled[/yellow]")
            else:
                rprint("[yellow]Warning:[/yellow] Source is outside cache/, not deleting for safety.")
                rprint(f"Source location: {source_path}")

        console.print(f"\n[green]✓[/green] Pipeline '{name}' removed successfully!")

    except Exception as err:
        rprint(f"[bold red]Error:[/bold red] Failed to remove pipeline: {err}")
        raise typer.Exit(code=1) from err


@app.command(name="validate")
def validate_pipeline(
    pipeline_name: str = typer.Argument(help="Name of the pipeline to validate"),
) -> None:
    """Validate a pipeline configuration.

    Checks that:
    - Pipeline module exists
    - Config file exists and is valid TOML
    - All referenced classes exist
    - Required sections are present

    Examples:
        pytorchpipeline validate sam2segnet
    """
    issues = []

    # Check pipeline module
    try:
        module = path_manager.import_project_module(pipeline_name)

        if not module:
            issues.append(f"✗ Pipeline module '{pipeline_name}' not found")
        else:
            permanences = getattr(module, "permanences_to_register", {})
            processes = getattr(module, "processes_to_register", {})

            if not permanences and not processes:
                issues.append("⚠ No permanences or processes registered")

            # Validate classes
            for name, cls in permanences.items():
                if not issubclass(cls, Permanence):
                    issues.append(f"✗ {name} is not a valid Permanence class")

            for name, cls in processes.items():
                if not issubclass(cls, PipelineProcess):
                    issues.append(f"✗ {name} is not a valid PipelineProcess class")

    except Exception as e:
        issues.append(f"✗ Error loading pipeline module: {e}")

    # Check config file
    config_path = path_manager.get_config_path(pipeline_name, "execute_pipeline.toml")
    if not config_path.exists():
        issues.append(f"✗ Config file not found: {config_path}")
    else:
        try:
            import tomllib

            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            # Check required sections
            if "permanences" not in config and "processes" not in config:
                issues.append("⚠ Config has neither permanences nor processes sections")

        except Exception as e:
            issues.append(f"✗ Invalid TOML config: {e}")

    # Display results
    if issues:
        console.print(
            Panel("\n".join(issues), title=f"[red]Validation Issues: {pipeline_name}[/red]", border_style="red")
        )
        raise typer.Exit(code=1)
    else:
        console.print(
            Panel(
                f"[green]✓[/green] Pipeline '{pipeline_name}' is valid!",
                title="Validation Passed",
                border_style="green",
            )
        )


@app.command(name="info")
def show_info() -> None:
    """Show information about the current PytorchImagePipeline installation.

    Displays the current operating mode (development vs production),
    directory paths for projects, configs, and cache, and other useful
    debugging information.

    Examples:
        pytorchpipeline info
    """
    info = path_manager.get_info()

    # Create table
    table = Table(title="PytorchImagePipeline Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    for key, value in info.items():
        # Format key nicely
        display_key = key.replace("_", " ").title()
        table.add_row(display_key, str(value))

    console.print(table)

    # Additional helpful info
    console.print("\n[bold]Directory Status:[/bold]")

    projects_dir = path_manager.get_projects_dir()
    configs_dir = path_manager.get_configs_dir()
    cache_dir = path_manager.get_cache_dir()

    status_items = [
        ("Projects Dir", projects_dir, projects_dir.exists()),
        ("Configs Dir", configs_dir, configs_dir.exists()),
        ("Cache Dir", cache_dir, cache_dir.exists()),
    ]

    for name, path, exists in status_items:
        status = "[green]✓[/green]" if exists else "[yellow]✗ (will be created)[/yellow]"
        console.print(f"  {status} {name}: {path}")


def _generate_project_init(project_name: str, with_example: bool) -> str:
    """Generate __init__.py for standalone project."""
    if with_example:
        return f'''"""Pipeline components for {project_name}."""

from {project_name}.permanences import ExamplePermanence
from {project_name}.processes import ExampleProcess

permanences_to_register = {{
    "ExamplePermanence": ExamplePermanence,
}}

processes_to_register = {{
    "ExampleProcess": ExampleProcess,
}}

__version__ = "0.1.0"
'''
    else:
        return f'''"""Pipeline components for {project_name}."""

# Import your permanences and processes here
# from {project_name}.permanences import MyPermanence
# from {project_name}.processes import MyProcess

permanences_to_register = {{
    # "MyPermanence": MyPermanence,
}}

processes_to_register = {{
    # "MyProcess": MyProcess,
}}

__version__ = "0.1.0"
'''


def _generate_init_file(with_example: bool) -> str:
    """Generate __init__.py content."""
    if with_example:
        return '''"""Pipeline components registration."""

from pytorchimagepipeline.pipelines.{pipeline_name}.permanence import ExamplePermanence
from pytorchimagepipeline.pipelines.{pipeline_name}.processes import ExampleProcess

permanences_to_register = {
    "ExamplePermanence": ExamplePermanence,
}

processes_to_register = {
    "ExampleProcess": ExampleProcess,
}
'''
    else:
        return '''"""Pipeline components registration."""

# Import your permanences and processes here
# from .permanence import MyPermanence
# from .processes import MyProcess

permanences_to_register = {
    # "MyPermanence": MyPermanence,
}

processes_to_register = {
    # "MyProcess": MyProcess,
}
'''


def _generate_permanence_file(pipeline_name: str, with_example: bool) -> str:
    """Generate permanence.py content."""
    base = f'''"""Permanence objects for {pipeline_name} pipeline.

Permanences are stateful objects that persist throughout pipeline execution.
"""

from typing import Optional

from pytorchimagepipeline.abstractions import Permanence


'''

    if with_example:
        base += '''class ExamplePermanence(Permanence):
    """Example permanence implementation.

    This is a template - replace with your actual permanence logic.
    """

    def __init__(self, config_value: str = "default"):
        """Initialize the permanence.

        Args:
            config_value: Example configuration parameter.
        """
        self.config_value = config_value
        self.data = None

    def initialize(self) -> Optional[Exception]:
        """Initialize resources before pipeline execution.

        Returns:
            Optional[Exception]: Error if initialization fails.
        """
        # Load data, allocate resources, etc.
        self.data = f"Initialized with {self.config_value}"
        return None

    def cleanup(self) -> Optional[Exception]:
        """Clean up resources after pipeline execution.

        Returns:
            Optional[Exception]: Error if cleanup fails.
        """
        # Release resources
        self.data = None
        return None
'''
    else:
        base += '''# Example permanence template:
#
# class MyPermanence(Permanence):
#     """Description of what this permanence manages."""
#
#     def __init__(self, param1: str, param2: int = 10):
#         self.param1 = param1
#         self.param2 = param2
#
#     def initialize(self) -> Optional[Exception]:
#         """Setup phase - called before processes run."""
#         # Initialize resources
#         return None
#
#     def cleanup(self) -> Optional[Exception]:
#         """Cleanup phase - called after processes complete."""
#         # Release resources
#         return None
'''

    return base


def _generate_processes_file(pipeline_name: str, with_example: bool) -> str:
    """Generate processes.py content."""
    base = f'''"""Process implementations for {pipeline_name} pipeline.

Processes are execution units that perform specific tasks.
"""

from typing import Optional

from pytorchimagepipeline.abstractions import PipelineProcess


'''

    if with_example:
        base += '''class ExampleProcess(PipelineProcess):
    """Example process implementation.

    This is a template - replace with your actual process logic.
    """

    def __init__(self, controller, force: bool = False, iterations: int = 10):
        """Initialize the process.

        Args:
            controller: Pipeline controller for accessing permanences.
            force: Force execution even if skipped.
            iterations: Number of iterations to perform.
        """
        self.controller = controller
        self.force = force
        self.iterations = iterations

    def execute(self) -> Optional[Exception]:
        """Execute the process logic.

        Returns:
            Optional[Exception]: Error if execution fails.
        """
        try:
            # Access permanences if needed
            # perm = self.controller.get_permanence("ExamplePermanence")

            # Your process logic here
            for i in range(self.iterations):
                print(f"Iteration {i+1}/{self.iterations}")

            return None
        except Exception as e:
            return e

    def skip(self) -> bool:
        """Check if this process should be skipped.

        Returns:
            bool: True if process should be skipped.
        """
        return False
'''
    else:
        base += '''# Example process template:
#
# class MyProcess(PipelineProcess):
#     """Description of what this process does."""
#
#     def __init__(self, controller, force: bool = False, param1: int = 10):
#         self.controller = controller
#         self.force = force
#         self.param1 = param1
#
#     def execute(self) -> Optional[Exception]:
#         """Execute the process."""
#         try:
#             # Access permanences
#             my_perm = self.controller.get_permanence("MyPermanence")
#
#             # Your logic here
#
#             return None
#         except Exception as e:
#             return e
#
#     def skip(self) -> bool:
#         """Check if process should be skipped."""
#         return False
'''

    return base


def _generate_readme(project_name: str) -> str:
    """Generate README.md for the project."""
    return f"""# {project_name}

Pipeline project for PytorchImagePipeline.

## Structure

```
{project_name}/
├── {project_name}/          # Package directory
│   ├── __init__.py         # Component registration
│   ├── permanences.py      # Permanence implementations
│   └── processes.py        # Process implementations
├── configs/                # Configuration files
│   └── pipeline_config.toml
├── pyproject.toml          # Package metadata
└── README.md
```

## Installation

This project can be used as a standalone package or linked to PytorchImagePipeline:

```bash
# Link to main pipeline
pytorchpipeline add /path/to/{project_name}
```

## Usage

```bash
# Validate configuration
pytorchpipeline validate {project_name}

# Run pipeline
pytorchpipeline run {project_name}
```

## Development

1. Edit `{project_name}/permanences.py` to add permanence objects
2. Edit `{project_name}/processes.py` to add process logic
3. Update `configs/pipeline_config.toml` with configuration
4. Test your pipeline with `pytorchpipeline validate {project_name}`

## License

Your license here
"""


def _generate_pyproject(project_name: str) -> str:
    """Generate pyproject.toml for the project."""
    return f"""[build-system]
requires = ["setuptools>=45", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "Pipeline project for PytorchImagePipeline"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "pytorchimagepipeline",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["{project_name}*"]

[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "ANN", "B", "A", "C4", "DTZ", "PYI", "PT"]
ignore = ["ANN101", "ANN102"]
"""


def _generate_gitignore() -> str:
    """Generate .gitignore for the project."""
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# OS
.DS_Store
Thumbs.db
"""


def _generate_config_file(pipeline_name: str, with_example: bool) -> str:
    """Generate execute_pipeline.toml content."""
    if with_example:
        return f"""# Configuration for {pipeline_name} pipeline

[permanences.example]
type = "ExamplePermanence"
params = {{ config_value = "my_value" }}

[processes.example]
type = "ExampleProcess"
params = {{ iterations = 5 }}
"""
    else:
        return f"""# Configuration for {pipeline_name} pipeline

# Define your permanences here:
# [permanences.my_permanence]
# type = "MyPermanence"
# params = {{ param1 = "value", param2 = 10 }}

# Define your processes here:
# [processes.my_process]
# type = "MyProcess"
# params = {{ param1 = 10 }}
"""


if __name__ == "__main__":
    app()
