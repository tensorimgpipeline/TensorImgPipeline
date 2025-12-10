"""CLI tool for TensorImgPipeline.

Provides commands for running, inspecting, and managing ML pipelines.

Copyright (C) 2025 Matti Kaupenjohann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from tipi.abstractions import Permanence, PipelineProcess
from tipi.core.runner import PipelineRunner
from tipi.paths import get_path_manager
from tipi.template_manager import ProjectSetup, template_manager


def _exit_with_error(message: str, code: int = 1, err: Exception | None = None) -> None:
    """Print error message and exit.

    Args:
        message: Error message to display
        code: Exit code (default: 1)
    """
    message = f"[bold red]Error:[/bold red][red] {message}[red]"
    rprint(message, file=sys.stderr)
    if err:
        raise typer.Exit(code=code) from err
    raise typer.Exit(code=code)


app = typer.Typer(
    name="tipi",
    help="TensorImgPipeline CLI - From scripts to production pipelines",
    add_completion=False,
)
console = Console()

# Get path manager
path_manager = get_path_manager()


@app.command(name="run")
def run_pipeline(
    pipeline_name: str = typer.Argument(help="The name of the pipeline to run (e.g., 'DemoFull')"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to custom config file (relative to configs/)"),
) -> None:
    """Run a pipeline by name.

    Examples:
        tipi run DemoFull
        tipi run DemoFull --config custom.toml
    """

    default_config = path_manager.get_config_path(pipeline_name)

    try:
        config_path = Path(config) if config else default_config
        runner = PipelineRunner(pipeline_name, config_path)
        runner.run()
    except Exception as err:
        _exit_with_error(str(err), err=err)


@app.command(name="list")
def list_pipelines(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    show_links: bool = typer.Option(True, "--links/--no-links", help="Show symlink information"),
) -> None:
    """List all available pipelines.

    Shows both built-in pipelines and linked subpackages.

    Examples:
        tipi list
        tipi list -v
        tipi list --no-links
    """
    pipelines_dir = path_manager.get_projects_dir()

    if not pipelines_dir.exists():
        _exit_with_error(
            f"[yellow]No pipelines directory found at {pipelines_dir}[/yellow]\n"
            f"Mode: {path_manager.get_info()['mode']}\n"
            "Create a pipeline with: [cyan]tipi create my_pipeline[/cyan]"
        )

    # Find all pipeline directories
    pipeline_dirs = [d for d in pipelines_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]

    if not pipeline_dirs:
        _exit_with_error("No pipelines found.")

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
        tipi inspect DemoFull
        tipi inspect DemoFull --docs
    """
    try:
        # Use path manager to import module
        module = path_manager.import_project_module(pipeline_name)
    except ImportError as err:
        _exit_with_error(f"Pipeline '{pipeline_name}' not found.\nCaused by Error: {err}")

    try:
        permanences = getattr(module, "permanences_to_register", {})
        processes = getattr(module, "processes_to_register", {})

        # Create tree view
        tree = Tree(f"[bold cyan]{pipeline_name}[/bold cyan] Pipeline")

        # Add permanences
        if permanences:
            perm_branch = tree.add("[bold green]Permanences[/bold green]")
            for permanece in permanences:
                cls_info = f"[cyan]{permanece.__name__}[/cyan]:"
                if show_docs and permanece.__doc__:
                    doc = permanece.__doc__.strip().split("\n")[0]
                    cls_info += f"\n  [dim]{doc}[/dim]"
                perm_branch.add(cls_info)
        else:
            tree.add("[yellow]No permanences registered[/yellow]")

        # Add processes
        if processes:
            proc_branch = tree.add("[bold magenta]Processes[/bold magenta]")
            for process in processes:
                cls_info = f"[magenta]{process.__name__}[/magenta]:"
                if show_docs and process.__doc__:
                    doc = process.__doc__.strip().split("\n")[0]
                    cls_info += f"\n  [dim]{doc}[/dim]"
                proc_branch.add(cls_info)
        else:
            tree.add("[yellow]No processes registered[/yellow]")

        console.print(tree)

        # Show config location
        config_path = path_manager.get_config_path(pipeline_name, "pipeline_config.toml")
        if config_path and config_path.exists():
            console.print(f"\n[green]✓[/green] Config: {config_path}")
        else:
            console.print(f"\n[red]✗[/red] Config not found: {config_path if config_path else 'N/A'}")

    except ModuleNotFoundError as err:
        _exit_with_error(f"Pipeline '{pipeline_name}' not found.", err=err)

    except Exception as err:
        _exit_with_error("", err=err)


@app.command(name="create")
def create_project(
    project_name: str = typer.Argument(help="Name of the new pipeline project"),
    location: str | None = typer.Option(
        None, "--location", "-l", help="Location to create project (default: current directory)"
    ),
    example: str = typer.Option("basic", "--example", "-e", help="Include working example process and permanence"),
    description: str | None = typer.Option(None, "--description", "-d", help="Project description"),
) -> None:
    """Create a new pipeline project with scaffolding.

    Creates a complete project structure for a new pipeline that can later
    be added to the main pipeline system.

    The --example flag creates a working pipeline with:
    - ConfigPermanence: For storing pipeline settings
    - DataPermanence: For storing data throughout pipeline
    - LoadDataProcess: Example process that loads data
    - ProcessDataProcess: Example process that processes data

    Examples:
        tipi create my_pipeline
        tipi create my_pipeline --location ./projects
        tipi create my_pipeline --example
        tipi create my_pipeline --example --description "My ML pipeline"
    """
    # Determine project location
    base_dir = Path(location) / project_name if location else Path.cwd() / project_name

    if base_dir.exists():
        _exit_with_error(f"Directory '{base_dir}' already exists.")

    # Validate example is one of the allowed values
    if example not in ("basic", "full", "pause"):
        _exit_with_error(f"Invalid example '{example}'. Must be one of: basic, full, pause")

    try:
        # Use template manager to create project
        # Cast is safe because we validated above
        project_data = ProjectSetup(
            name=project_name,
            base_dir=base_dir,
            example=example,  # type: ignore[arg-type]
            description=description,
        )

        template_manager.create_project(project_data=project_data)
    except Exception as err:
        _exit_with_error("Failed to create projec", err=err)

    # Success message
    example_note = " with working example" if example != "basic" else ""
    panel = Panel(
        f"[green]✓[/green] Pipeline project '{project_name}' created successfully{example_note}!\n\n"
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
        + (
            f"  2. Review the example code in {project_name}/\n"
            f"  3. Link to main pipeline: tipi add {base_dir}\n"
            f"  4. Run the pipeline: tipi run {project_name}"
            if example != "basic"
            else f"  2. Edit {project_name}/permanences.py and processes.py\n"
            f"  3. Update configs/pipeline_config.toml\n"
            f"  4. Link to main pipeline: tipi add {base_dir}"
        ),
        title="Project Created",
        border_style="green",
    )
    console.print(panel)


@app.command(name="add")
def add_subpackage(
    source: str = typer.Argument(help="Local path to project or Git repository URL"),
    name: str | None = typer.Option(
        None, "--name", "-n", help="Name for the linked pipeline (defaults to directory name)"
    ),
    location: str | None = typer.Option(
        None, "--location", "-l", help="Location to clone git repos (default: cache_dir/submodules)"
    ),
    branch: str | None = typer.Option(None, "--branch", "-b", help="Git branch to checkout (if cloning)"),
) -> None:
    """Link an existing project or clone from git as a subpackage.

    This command either:
    1. Links a local project directory to the pipeline system
    2. Clones a git repository and links it

    Examples:
        # Link local project
        tipi add ./my_pipeline_project
        tipi add /path/to/project --name custom_name

        # Clone and link from git
        tipi add https://github.com/user/ml-pipeline.git
        tipi add git@github.com:user/pipeline.git --location ./external
        tipi add https://github.com/user/repo.git --branch dev
    """

    try:
        # Check if source is a git URL
        is_git_url = source.startswith(("https://", "http://", "git@", "git://"))

        if is_git_url:
            # Clone from git - use cache directory for isolation
            clone_location = Path(location) if location else path_manager.get_cache_dir() / "projects"
            clone_location.mkdir(parents=True, exist_ok=True)

            # Extract repo name from URL if name not provided
            if not name:
                name = Path(source.rstrip("/")).stem.replace(".git", "")

            target_dir = clone_location / name

            if target_dir.exists():
                _exit_with_error(f"Target directory '{target_dir}' already exists.")

            console.print("[cyan]Cloning repository...[/cyan]")

            # Build git clone command
            cmd = ["git", "clone"]
            if branch:
                cmd.extend(["--branch", branch])
            cmd.extend([source, str(target_dir)])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                _exit_with_error(f"Git clone failed:\n{result.stderr}")

            console.print(f"[green]✓[/green] Repository cloned to {target_dir}")
            project_path = target_dir

        else:
            # Link local project
            project_path = Path(source).resolve()

            if not project_path.exists():
                _exit_with_error(f"Path '{source}' does not exist.")

            if not project_path.is_dir():
                _exit_with_error(f"Path '{source}' is not a directory.")

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
            _exit_with_error("Not a valid Python package (missing __init__.py)")

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
                _exit_with_error(f"A directory/file already exists at {link_path}")

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
            f"  1. Inspect: tipi inspect {name}\n"
            f"  2. Validate: tipi validate {name}\n"
            f"  3. Run: tipi run {name}",
            title="Package Added",
            border_style="green",
        )
        console.print(panel)

    except subprocess.CalledProcessError as err:
        _exit_with_error("Command failed.", err=err)
    except Exception as err:
        _exit_with_error("", err=err)


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
        tipi remove my_pipeline
        tipi remove my_pipeline --delete-source
    """
    projects_dir = path_manager.get_projects_dir()
    link_path = projects_dir / name

    if not link_path.exists():
        _exit_with_error(f"Pipeline '{name}' not found.")

    if not link_path.is_symlink():
        _exit_with_error(
            f"'{name}' is not a linked package (it's built-in).\nBuilt-in pipelines cannot be removed this way."
        )

    try:
        # Get the source path before unlinking
        source_path = link_path.resolve().parent

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
            # checking for relative position to cache is fine,
            # since only git repo will be stored in cache.
            if source_path.is_relative_to(cache_dir):
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
        _exit_with_error("Failed to remove pipeline.", err=err)


@app.command(name="clean")
def clean_broken_symlinks(
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be removed without actually removing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
) -> None:
    """Check and remove broken symlinks in projects and configs.

    Scans the projects and configs directories for broken symlinks
    (links pointing to non-existent targets) and removes them.

    Examples:
        tipi clean                    # Remove broken symlinks
        tipi clean --dry-run          # Preview what would be removed
        tipi clean --verbose          # Show details of all symlinks
    """
    projects_dir = path_manager.get_projects_dir()
    configs_dir = path_manager.get_configs_dir()

    broken_links = []
    valid_links = []

    # Check projects directory
    if projects_dir.exists():
        for item in projects_dir.iterdir():
            if item.is_symlink():
                target = item.resolve(strict=False)
                if not target.exists():
                    broken_links.append(("project", item, target))
                else:
                    valid_links.append(("project", item, target))

    # Check configs directory
    if configs_dir.exists():
        for item in configs_dir.iterdir():
            if item.is_symlink():
                target = item.resolve(strict=False)
                if not target.exists():
                    broken_links.append(("config", item, target))
                else:
                    valid_links.append(("config", item, target))

    # Display results
    if verbose and valid_links:
        console.print("\n[green]Valid symlinks:[/green]")
        for link_type, link_path, target in valid_links:
            console.print(f"  [green]✓[/green] {link_type:8} {link_path.name} → {target}")

    if not broken_links:
        console.print("\n[green]✓[/green] No broken symlinks found!")
        return

    # Display broken symlinks
    console.print(f"\n[yellow]Found {len(broken_links)} broken symlink(s):[/yellow]")
    for link_type, link_path, target in broken_links:
        console.print(f"  [red]✗[/red] {link_type:8} {link_path.name} → [red]{target}[/red]")

    if dry_run:
        console.print("\n[cyan]Dry run:[/cyan] No changes made. Run without --dry-run to remove.")
        return

    # Remove broken symlinks
    console.print()
    removed_count = 0
    for _link_type, link_path, _target in broken_links:
        try:
            link_path.unlink()
            console.print(f"[green]✓[/green] Removed: {link_path.name}")
            removed_count += 1
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to remove {link_path.name}: {e}")

    console.print(f"\n[green]✓[/green] Cleaned up {removed_count}/{len(broken_links)} broken symlink(s)!")


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
        tipi validate DemoFull
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
    config_path = path_manager.get_config_path(pipeline_name, "pipeline_config.toml")
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
    """Show information about the current TensorImgPipeline installation.

    Displays the current operating mode (development vs production),
    directory paths for projects, configs, and cache, and other useful
    debugging information.

    Examples:
        tipi info
    """
    info = path_manager.get_info()

    # Create table
    table = Table(title="TensorImgPipeline Configuration", show_header=True, header_style="bold magenta")
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


if __name__ == "__main__":
    app()
